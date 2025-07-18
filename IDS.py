import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from scapy.all import sniff, IP, TCP, UDP
from collections import deque, defaultdict
from threading import Thread, RLock

# --- Configuration ---
FLOW_TIMEOUT = 120  # seconds to wait before a flow is considered inactive
HISTORY_MAXLEN = 500 # Max number of recent connections to store for feature calculation
LOG_PREFIX = "[IDS]"

# --- Color Codes for Alerts ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Load Model and Preprocessing Artifacts ---
print(f"{bcolors.HEADER}{LOG_PREFIX} Loading model and preprocessing artifacts...{bcolors.ENDC}")
try:
    model = tf.keras.models.load_model('ids_final_model.keras')
    scaler = joblib.load('scaler.gz')
    model_columns = joblib.load('model_columns.pkl')
    # Load the optimized threshold calculated during training
    best_threshold = joblib.load('best_threshold.pkl') 
    print(f"{bcolors.OKGREEN}{LOG_PREFIX} Artifacts loaded successfully.{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{LOG_PREFIX} Using optimal prediction threshold: {best_threshold:.4f}{bcolors.ENDC}")
except FileNotFoundError as e:
    print(f"{bcolors.FAIL}{LOG_PREFIX} Error: Could not load required file. {e}{bcolors.ENDC}")
    print(f"{bcolors.FAIL}{LOG_PREFIX} Please run AITrain.py to generate the required model and artifact files.{bcolors.ENDC}")
    exit()
except Exception as e:
    print(f"{bcolors.FAIL}{LOG_PREFIX} An unexpected error occurred during artifact loading: {e}{bcolors.ENDC}")
    exit()

# --- Global State with Thread-Safe Locks ---
active_flows = {}
connection_history = deque(maxlen=HISTORY_MAXLEN)
global_lock = RLock()

# --- Expanded Service Port Mapping (from UNSW-NB15 dataset) ---
service_map = {
    80: 'http', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
    443: 'https', 110: 'pop3', 143: 'imap', 123: 'ntp', 3389: 'rdp',
    137: 'netbios-ns', 139: 'netbios-ssn', 445: 'microsoft-ds',
    67: 'dhcp', 68: 'dhcp', 161: 'snmp', 162: 'snmp',
    # Common ports from dataset services file
    5060: 'sip', 389: 'ldap', 993: 'imaps', 995: 'pop3s', 5222: 'xmpp-client'
}

def get_flow_id(packet):
    """Creates a unique, direction-agnostic ID for a network flow."""
    if IP in packet and (TCP in packet or UDP in packet):
        protocol = 'tcp' if TCP in packet else 'udp'
        sport = packet[protocol].sport
        dport = packet[protocol].dport
        return tuple(sorted(((packet[IP].src, sport), (packet[IP].dst, dport))))
    return None

def extract_features(flow_data, now):
    """Extracts features from flow data and approximates host/time-based metrics."""
    features = defaultdict(float)

    # Basic features from flow
    features['dur'] = flow_data['duration']
    features['sbytes'] = flow_data['src_bytes']
    features['dbytes'] = flow_data['dst_bytes']
    features['land'] = 1 if flow_data['src_ip'] == flow_data['dst_ip'] else 0
    
    # One-hot encode protocol and service
    service = flow_data['service']
    proto = flow_data['protocol']
    features[f'proto_{proto}'] = 1.0
    features[f'service_{service}'] = 1.0

    # One-hot encode state based on flags
    # This is a simplified mapping from TCP flags to UNSW-NB15 'state'
    flags = flow_data['flags']
    state = "FIN" if 'F' in flags else "RST" if 'R' in flags else "CON" if 'S' in flags and 'A' in flags else "REQ" if 'S' in flags else "INT"
    features[f'state_{state}'] = 1.0
    
    # Time-based and host-based feature approximations
    with global_lock:
        recent_conns_2s = [c for c in connection_history if now - c['timestamp'] <= 2]
        dst_host_conns_100 = [c for c in connection_history if c['dst_host'] == flow_data['dst_ip']]
    
    # `count` and `srv_count` (Connections to same host/service in last 2s)
    features['count'] = sum(1 for c in recent_conns_2s if c['dst_host'] == flow_data['dst_ip'])
    features['srv_count'] = sum(1 for c in recent_conns_2s if c['dst_host'] == flow_data['dst_ip'] and c['service'] == service)

    # `dst_host_count` and `dst_host_srv_count` (Connections to same host/service in last 100 conns)
    features['dst_host_count'] = len(dst_host_conns_100)
    features['dst_host_srv_count'] = sum(1 for c in dst_host_conns_100 if c['service'] == service)

    # `dst_host_same_srv_rate`
    if features['dst_host_count'] > 0:
        features['dst_host_same_srv_rate'] = features['dst_host_srv_count'] / features['dst_host_count']
    
    # `dst_host_diff_srv_rate`
    if features['dst_host_count'] > 0:
        diff_srv_count = sum(1 for c in dst_host_conns_100 if c['service'] != service)
        features['dst_host_diff_srv_rate'] = diff_srv_count / features['dst_host_count']

    # `dst_host_same_src_port_rate`
    if features['dst_host_count'] > 0:
        same_src_port_count = sum(1 for c in dst_host_conns_100 if c['src_port'] == flow_data['src_port'])
        features['dst_host_same_src_port_rate'] = same_src_port_count / features['dst_host_count']

    return features

def process_and_predict(flow_data):
    """Processes a completed flow, predicts its class, and alerts if necessary."""
    now = time.time()
    
    # 1. Feature Extraction
    features = extract_features(flow_data, now)
    live_df = pd.DataFrame([features])
    
    # 2. Preprocessing
    # Align columns with the model's training data
    live_df_aligned = live_df.reindex(columns=model_columns, fill_value=0.0)
    
    # Identify numerical columns for scaling (that are present in the final features)
    numerical_cols = [col for col in model_columns if live_df_aligned[col].dtype != 'object' and not col.startswith(('proto_', 'service_', 'state_'))]
    
    # Ensure scaler only transforms columns it was fitted on and are present
    cols_to_scale = [col for col in numerical_cols if col in live_df_aligned.columns]
    if cols_to_scale:
        live_df_aligned[cols_to_scale] = scaler.transform(live_df_aligned[cols_to_scale])

    # 3. Prediction
    try:
        prediction_prob = model.predict(live_df_aligned, verbose=0)[0][0]
        is_malicious = prediction_prob > best_threshold
    except Exception as e:
        print(f"{bcolors.WARNING}{LOG_PREFIX} Prediction error: {e}{bcolors.ENDC}")
        return

    # 4. Alerting
    if is_malicious:
        alert_message = (
            f"\n{bcolors.FAIL}{bcolors.BOLD}================[ MALICIOUS ACTIVITY DETECTED ]================{bcolors.ENDC}\n"
            f"{bcolors.FAIL} Flow            : {flow_data['src_ip']}:{flow_data['src_port']} -> {flow_data['dst_ip']}:{flow_data['dst_port']}\n"
            f" Protocol        : {flow_data['protocol'].upper()} ({flow_data['service']})\n"
            f" Duration        : {flow_data['duration']:.4f}s\n"
            f" Confidence      : {prediction_prob:.2%}\n"
            f" Classification  : MALICIOUS (Threshold: {best_threshold:.2f})\n"
            f"==============================================================={bcolors.ENDC}"
        )
        print(alert_message)
    
    # 5. Update History
    with global_lock:
        connection_history.append({
            'timestamp': now,
            'dst_host': flow_data['dst_ip'],
            'src_port': flow_data['src_port'],
            'service': flow_data['service'],
        })

def packet_callback(packet):
    """Callback function for each captured packet."""
    now = time.time()
    flow_id = get_flow_id(packet)
    if not flow_id: return

    with global_lock:
        if flow_id not in active_flows:
            proto = 'tcp' if TCP in packet else 'udp'
            active_flows[flow_id] = {
                'start_time': now, 'last_time': now,
                'src_ip': packet[IP].src, 'dst_ip': packet[IP].dst,
                'src_port': packet[proto].sport, 'dst_port': packet[proto].dport,
                'protocol': proto, 'src_bytes': 0, 'dst_bytes': 0,
                'flags': set(),
                'service': service_map.get(packet[proto].dport, service_map.get(packet[proto].sport, 'others'))
            }
        
        flow = active_flows[flow_id]
        flow['last_time'] = now
        
        # Determine direction for byte count
        if packet[IP].src == flow['src_ip']:
            flow['src_bytes'] += len(packet.payload)
        else:
            flow['dst_bytes'] += len(packet.payload)
            
        if TCP in packet:
            # Using scapy's string representation of flags (e.g., 'S' for SYN, 'F' for FIN)
            for flag in str(packet[TCP].flags):
                flow['flags'].add(flag)

        # Trigger analysis on flow termination (FIN or RST)
        if TCP in packet and (packet[TCP].flags.F or packet[TCP].flags.R):
            flow_to_process = active_flows.pop(flow_id)
            flow_to_process['duration'] = flow_to_process['last_time'] - flow_to_process['start_time']
            # Run prediction in a new thread to avoid blocking the sniffer
            Thread(target=process_and_predict, args=(flow_to_process,)).start()

def flow_manager():
    """A thread to manage and timeout inactive flows."""
    while True:
        time.sleep(15)
        now = time.time()
        with global_lock:
            timed_out_ids = [fid for fid, flow in active_flows.items() if now - flow['last_time'] > FLOW_TIMEOUT]
            
            for fid in timed_out_ids:
                flow_data = active_flows.pop(fid, None)
                if flow_data:
                    flow_data['duration'] = flow_data['last_time'] - flow_data['start_time']
                    print(f"{bcolors.OKCYAN}{LOG_PREFIX} Flow timed out. Analyzing {flow_data['src_ip']} -> {flow_data['dst_ip']}...{bcolors.ENDC}")
                    # Run prediction in a new thread
                    Thread(target=process_and_predict, args=(flow_data,)).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced AI-based Network Intrusion Detection System")
    parser.add_argument('--interface', type=str, required=True, help='Network interface to sniff on (e.g., eth0, wlan0)')
    args = parser.parse_args()

    print(f"\n{bcolors.BOLD}--- Starting Live Network Traffic Analyzer ---{bcolors.ENDC}")
    print(f"{LOG_PREFIX} Sniffing on interface: {bcolors.OKGREEN}{args.interface}{bcolors.ENDC}")
    print(f"{LOG_PREFIX} Press {bcolors.WARNING}Ctrl+C{bcolors.ENDC} to stop.")

    manager = Thread(target=flow_manager, daemon=True)
    manager.start()

    try:
        sniff(iface=args.interface, prn=packet_callback, store=0)
    except PermissionError:
        print(f"\n{bcolors.FAIL}{LOG_PREFIX} Permission denied. Please run this script with 'sudo'.{bcolors.ENDC}")
    except OSError as e:
        print(f"\n{bcolors.FAIL}{LOG_PREFIX} Error sniffing on interface '{args.interface}': {e}{bcolors.ENDC}")
        print(f"{bcolors.WARNING}{LOG_PREFIX} Please ensure the interface name is correct and the interface is up.{bcolors.ENDC}")
    except Exception as e:
        print(f"\n{bcolors.FAIL}{LOG_PREFIX} An unexpected error occurred: {e}{bcolors.ENDC}")
    finally:
        print(f"\n{bcolors.BOLD}--- Analyzer stopped ---{bcolors.ENDC}")
