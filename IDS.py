# live_ids_analyzer.py

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from scapy.all import sniff, IP, TCP, UDP
from collections import deque, defaultdict
from threading import Thread

# --- Load Model and Preprocessing Artifacts ---
print("--- Loading model and preprocessing artifacts ---")
try:
    model = tf.keras.models.load_model('ids_model.h5')
    scaler = joblib.load('scaler.gz')
    model_columns = joblib.load('model_columns.pkl')
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not load required file. {e}")
    print("Please ensure 'ids_model.h5', 'scaler.gz', and 'model_columns.pkl' are in the same directory.")
    exit()

# --- Global State for Flow Tracking ---
active_flows = {}  # Key: flow_id, Value: flow data
connection_history = deque(maxlen=100) # Stores feature dicts of recent connections
FLOW_TIMEOUT = 60  # seconds to wait before a flow is considered inactive

# --- Service Port Mapping (Simplified) ---
service_map = {
    80: 'http', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain_u', 
    443: 'http_443', 110: 'pop_3', 143: 'imap4'
}

def get_flow_id(packet):
    """Creates a unique ID for a network flow."""
    if IP in packet:
        return tuple(sorted(((packet[IP].src, packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else 0),
                             (packet[IP].dst, packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else 0))))
    return None

def process_and_predict(flow_data):
    """Extracts features, preprocesses, and predicts using the model."""
    global connection_history

    # --- 1. Basic Feature Extraction ---
    features = {
        'duration': flow_data['duration'],
        'src_bytes': flow_data['src_bytes'],
        'dst_bytes': flow_data['dst_bytes'],
        'land': 1 if flow_data['src_ip'] == flow_data['dst_ip'] and flow_data['src_port'] == flow_data['dst_port'] else 0,
        'wrong_fragment': 0, # Simplified
        'urgent': 0, # Simplified
    }
    
    # --- 2. Approximated Time-based and Host-based Features ---
    now = time.time()
    recent_2s = [conn for conn in connection_history if now - conn['timestamp'] <= 2]
    
    features['count'] = sum(1 for conn in recent_2s if conn['dst_host'] == flow_data['dst_ip'])
    features['srv_count'] = sum(1 for conn in recent_2s if conn['dst_host'] == flow_data['dst_ip'] and conn['service'] == flow_data['service'])

    serror_count = sum(1 for conn in recent_2s if conn['dst_host'] == flow_data['dst_ip'] and 'S0' in conn['flag'])
    srv_serror_count = sum(1 for conn in recent_2s if conn['dst_host'] == flow_data['dst_ip'] and conn['service'] == flow_data['service'] and 'S0' in conn['flag'])
    
    features['serror_rate'] = serror_count / len(recent_2s) if len(recent_2s) > 0 else 0
    features['srv_serror_rate'] = srv_serror_count / len(recent_2s) if len(recent_2s) > 0 else 0
    
    for col in model_columns:
        if col not in features and col in ['rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                                           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                                           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']:
            features[col] = 0.0
        elif col not in features:
            features[col] = 0

    features['protocol_type_' + flow_data['protocol']] = 1
    features['service_' + flow_data['service']] = 1
    features['flag_' + "".join(sorted(list(flow_data['flags'])))] = 1
    
    # --- 3. Preprocessing ---
    live_df = pd.DataFrame([features])
    live_df_aligned = live_df.reindex(columns=model_columns, fill_value=0)
    
    numerical_cols = [col for col in model_columns if live_df_aligned[col].dtype != 'object' and not col.startswith(('protocol_type_', 'service_', 'flag_'))]
    live_df_aligned[numerical_cols] = scaler.transform(live_df_aligned[numerical_cols])

    # --- 4. Prediction ---
    prediction_prob = model.predict(live_df_aligned, verbose=0)[0][0]
    prediction = 'MALICIOUS' if prediction_prob > 0.5 else 'Normal'

    # --- 5. Alerting ---
    if prediction == 'MALICIOUS':
        print(f"\033[91mALERT! {prediction} traffic detected! (Confidence: {prediction_prob:.2%})\033[0m")
        print(f"  Flow: {flow_data['src_ip']}:{flow_data['src_port']} -> {flow_data['dst_ip']}:{flow_data['dst_port']} ({flow_data['protocol']})")
    
    # --- 6. Update History ---
    connection_history.append({
        'timestamp': now,
        'dst_host': flow_data['dst_ip'],
        'service': flow_data['service'],
        'flag': "".join(sorted(list(flow_data['flags'])))
    })

def packet_callback(packet):
    """Callback function for each captured packet."""
    if not (IP in packet and (TCP in packet or UDP in packet)):
        return

    flow_id = get_flow_id(packet)
    if not flow_id:
        return

    now = time.time()
    
    if flow_id not in active_flows:
        active_flows[flow_id] = {
            'start_time': now, 'last_time': now, 'src_ip': packet[IP].src,
            'dst_ip': packet[IP].dst, 'src_port': packet.sport, 'dst_port': packet.dport,
            'protocol': 'tcp' if TCP in packet else 'udp', 'src_bytes': 0, 'dst_bytes': 0,
            'flags': set(),
            'service': service_map.get(packet.dport, 'other')
        }
    
    flow = active_flows[flow_id]
    flow['last_time'] = now
    
    if packet[IP].src == flow['src_ip']:
        flow['src_bytes'] += len(packet)
    else:
        flow['dst_bytes'] += len(packet)
        
    if TCP in packet:
        flow['flags'].add(str(packet[TCP].flags))

    if TCP in packet and (packet[TCP].flags.F or packet[TCP].flags.R):
        flow['duration'] = flow['last_time'] - flow['start_time']
        process_and_predict(flow)
        del active_flows[flow_id]

def flow_manager():
    """A thread to manage and timeout inactive flows."""
    while True:
        time.sleep(10)
        now = time.time()
        timed_out_flows = []
        for flow_id, flow_data in list(active_flows.items()):
            if now - flow_data['last_time'] > FLOW_TIMEOUT:
                timed_out_flows.append(flow_id)
        
        for flow_id in timed_out_flows:
            flow_data = active_flows.pop(flow_id, None)
            if flow_data:
                flow_data['duration'] = flow_data['last_time'] - flow_data['start_time']
                print(f"Flow timed out. Analyzing {flow_data['src_ip']} -> {flow_data['dst_ip']}...")
                process_and_predict(flow_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI-based Network Intrusion Detection System")
    parser.add_argument('--interface', type=str, required=True, help='Network interface to sniff on (e.g., eth0)')
    args = parser.parse_args()

    print("--- Starting Live Network Traffic Analyzer ---")
    print(f"Sniffing on interface: {args.interface}")
    print("Press Ctrl+C to stop.")

    manager_thread = Thread(target=flow_manager, daemon=True)
    manager_thread.start()

    try:
        sniff(iface=args.interface, prn=packet_callback, store=0)
    except PermissionError:
        print("\nError: Permission denied. Please run this script with sudo.")
    except OSError as e:
        print(f"\nError sniffing on interface {args.interface}: {e}")
        print("Please ensure the interface exists and is correct.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n--- Analyzer stopped ---")
