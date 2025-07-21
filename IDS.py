# IDS.py - Final Hardened Version with Pre-trained Flag for Jetson Nano

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import sys
import os
from collections import deque, defaultdict
from threading import Thread, RLock
from scapy.config import conf
# --- Scapy Imports ---
from scapy.all import sniff, IP, TCP, UDP, L3RawSocket

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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Advanced AI-based Network Intrusion Detection System")
# Hint for Jetson Nano users: common interfaces are eth0, wlan0
parser.add_argument('--interface', type=str, required=True, help='Network interface to sniff on (e.g., "eth0", "wlan0")')
parser.add_argument('--use-pretrained', action='store_true', help='Use the pre-trained model from the "pretrained" folder.')
args = parser.parse_args()

# --- Load Model and Preprocessing Artifacts ---
# Determine the path to the model files based on the flag
if args.use_pretrained:
    model_dir = 'pretrained'
    print(f"{bcolors.HEADER}{LOG_PREFIX} Loading PRE-TRAINED model and artifacts from '{model_dir}/'...{bcolors.ENDC}")
else:
    model_dir = ''
    print(f"{bcolors.HEADER}{LOG_PREFIX} Loading self-trained model and artifacts from the project root...{bcolors.ENDC}")

try:
    model_path = os.path.join(model_dir, 'ids_final_model.keras')
    scaler_path = os.path.join(model_dir, 'scaler.gz')
    columns_path = os.path.join(model_dir, 'model_columns.pkl')
    threshold_path = os.path.join(model_dir, 'best_threshold.pkl')

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    model_columns = joblib.load(columns_path)
    best_threshold = joblib.load(threshold_path)
    
    print(f"{bcolors.OKGREEN}{LOG_PREFIX} Artifacts loaded successfully.{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{LOG_PREFIX} Using optimal prediction threshold: {best_threshold:.4f}{bcolors.ENDC}")

except FileNotFoundError as e:
    print(f"{bcolors.FAIL}{LOG_PREFIX} Error: Could not load required file. {e}{bcolors.ENDC}")
    if args.use_pretrained:
        print(f"{bcolors.FAIL}{LOG_PREFIX} Ensure the 'pretrained' folder exists and contains all four model files.{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}{LOG_PREFIX} Please run AITrain.py to generate the required model and artifact files in the root directory.{bcolors.ENDC}")
    exit()
except Exception as e:
    print(f"{bcolors.FAIL}{LOG_PREFIX} An unexpected error occurred during artifact loading: {e}{bcolors.ENDC}")
    exit()

# --- Global State with Thread-Safe Locks ---
active_flows = {}
connection_history = deque(maxlen=HISTORY_MAXLEN)
global_lock = RLock()

# --- Expanded Service Port Mapping ---
service_map = {
    80: 'http', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
    443: 'https', 110: 'pop3', 143: 'imap', 123: 'ntp', 3389: 'rdp',
    137: 'netbios-ns', 139: 'netbios-ssn', 445: 'microsoft-ds', 67: 'dhcp',
    161: 'snmp', 5060: 'sip', 389: 'ldap', 993: 'imaps', 995: 'pop3s'
}

def get_flow_id(packet):
    # This function remains the same as it operates on parsed Scapy packet layers
    if IP in packet and (TCP in packet or UDP in packet):
        protocol = 'tcp' if TCP in packet else 'udp'
        # Sort tuple to ensure flow_id is consistent regardless of src/dst order
        return tuple(sorted(((packet[IP].src, packet[protocol].sport), (packet[IP].dst, packet[protocol].dport))))
    return None

def extract_features(flow_data, now):
    # This function remains the same as it operates on extracted flow_data
    features = defaultdict(float)
    features['dur'] = flow_data['duration']
    features['sbytes'] = flow_data['src_bytes']
    features['dbytes'] = flow_data['dst_bytes']
    features['land'] = 1 if flow_data['src_ip'] == flow_data['dst_ip'] else 0
    service = flow_data['service']
    proto = flow_data['protocol']
    features[f'proto_{proto}'] = 1.0
    features[f'service_{service}'] = 1.0
    flags = flow_data['flags']
    state = "FIN" if 'F' in flags else "RST" if 'R' in flags else "CON" if 'S' in flags and 'A' in flags else "REQ" if 'S' in flags else "INT"
    features[f'state_{state}'] = 1.0

    with global_lock:
        # Calculate features based on recent connection history
        recent_conns_2s = [c for c in connection_history if now - c['timestamp'] <= 2]
        dst_host_conns_100 = [c for c in connection_history if c['dst_host'] == flow_data['dst_ip']]

    features['count'] = sum(1 for c in recent_conns_2s if c['dst_host'] == flow_data['dst_ip'])
    features['srv_count'] = sum(1 for c in recent_conns_2s if c['dst_host'] == flow_data['dst_ip'] and c['service'] == service)
    features['dst_host_count'] = len(dst_host_conns_100)
    features['dst_host_srv_count'] = sum(1 for c in dst_host_conns_100 if c['service'] == service)

    if features['dst_host_count'] > 0:
        features['dst_host_same_srv_rate'] = features['dst_host_srv_count'] / features['dst_host_count']
        diff_srv_count = sum(1 for c in dst_host_conns_100 if c['service'] != service)
        features['dst_host_diff_srv_rate'] = diff_srv_count / features['dst_host_count']
        same_src_port_count = sum(1 for c in dst_host_conns_100 if c['src_port'] == flow_data['src_port'])
        features['dst_host_same_src_port_rate'] = same_src_port_count / features['dst_host_count']
    return features

def process_and_predict(flow_data):
    # This function remains the same as it processes flow_data and interacts with the model
    now = time.time()
    features = extract_features(flow_data, now)
    live_df = pd.DataFrame([features])
    live_df_aligned = live_df.reindex(columns=model_columns, fill_value=0.0)
    
    # Identify numerical columns for scaling
    numerical_cols = [col for col in model_columns if not col.startswith(('proto_', 'service_', 'state_'))]
    cols_to_scale = [col for col in numerical_cols if col in live_df_aligned.columns]
    
    if cols_to_scale:
        live_df_aligned[cols_to_scale] = scaler.transform(live_df_aligned[cols_to_scale])
    
    try:
        prediction_prob = model.predict(live_df_aligned, verbose=0)[0][0]
        is_malicious = prediction_prob > best_threshold
    except Exception as e:
        print(f"{bcolors.WARNING}{LOG_PREFIX} Prediction error: {e}{bcolors.ENDC}")
        return
    
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
    
    with global_lock:
        # Add connection to history for rate-based feature calculation
        connection_history.append({'timestamp': now, 'dst_host': flow_data['dst_ip'], 'src_port': flow_data['src_port'], 'service': flow_data['service']})

def packet_callback(packet):
    """
    Callback function for sniff().
    Processes each captured packet to extract flow data and trigger prediction.
    Now includes a try-except block to gracefully handle malformed packets.
    """
    try:
        # Check for IP layer, as L3RawSocket only captures IP packets
        if not packet.haslayer(IP):
            return

        proto = None
        if packet.haslayer(TCP):
            proto = 'tcp'
        elif packet.haslayer(UDP):
            proto = 'udp'
        else:
            # If it's an IP packet but neither TCP nor UDP (e.g., ICMP), skip for now
            # You could extend this to handle other protocols if your model supports them
            return

        now = time.time()
        flow_id = get_flow_id(packet)
        if not flow_id:
            return # Should not happen if IP, TCP/UDP are present

        with global_lock:
            # Initialize or update flow data
            if flow_id not in active_flows:
                active_flows[flow_id] = {
                    'start_time': now,
                    'src_ip': packet[IP].src,
                    'dst_ip': packet[IP].dst,
                    'src_port': packet[proto].sport,
                    'dst_port': packet[proto].dport,
                    'protocol': proto,
                    'src_bytes': 0,
                    'dst_bytes': 0,
                    'flags': set(),
                    'service': service_map.get(packet[proto].dport, service_map.get(packet[proto].sport, 'others'))
                }

            flow = active_flows[flow_id]
            flow['last_time'] = now

            # Update byte counts
            if packet[IP].src == flow['src_ip']:
                flow['src_bytes'] += len(packet.payload)
            else:
                flow['dst_bytes'] += len(packet.payload)

            # Update TCP flags if applicable
            if proto == 'tcp' and TCP in packet: # Ensure it's TCP before accessing flags
                for flag in str(packet[TCP].flags):
                    flow['flags'].add(flag)
                
                # If TCP connection is finishing (FIN or RST flag), process the flow
                if packet[TCP].flags.F or packet[TCP].flags.R:
                    flow_to_process = active_flows.pop(flow_id, None)
                    if flow_to_process:
                        flow_to_process['duration'] = flow_to_process['last_time'] - flow_to_process['start_time']
                        Thread(target=process_and_predict, args=(flow_to_process,)).start()
    except Exception as e:
        # Catch any unexpected errors during packet processing to prevent sniffer from crashing
        print(f"{bcolors.WARNING}{LOG_PREFIX} Skipping malformed or unprocessable packet: {e}{bcolors.ENDC}")

def flow_manager():
    """Manages and times out inactive flows."""
    while True:
        time.sleep(15) # Check for timeouts every 15 seconds
        now = time.time()
        with global_lock:
            # Identify flows that have been inactive for longer than FLOW_TIMEOUT
            timed_out_ids = [fid for fid, flow in active_flows.items() if now - flow['last_time'] > FLOW_TIMEOUT]
            for fid in timed_out_ids:
                flow_data = active_flows.pop(fid, None) # Remove from active flows
                if flow_data:
                    # Calculate duration for timed-out flows
                    flow_data['duration'] = flow_data['last_time'] - flow_data['start_time']
                    print(f"{bcolors.OKCYAN}{LOG_PREFIX} Flow timed out. Analyzing {flow_data['src_ip']} -> {flow_data['dst_ip']}...{bcolors.ENDC}")
                    # Process and predict in a separate thread to avoid blocking
                    Thread(target=process_and_predict, args=(flow_data,)).start()

if __name__ == '__main__':
    print(f"\n{bcolors.BOLD}--- Starting Live Network Traffic Analyzer ---{bcolors.ENDC}")
    print(f"{LOG_PREFIX} Sniffing on interface: {bcolors.OKGREEN}{args.interface}{bcolors.ENDC}")
    print(f"{LOG_PREFIX} Press {bcolors.WARNING}Ctrl+C{bcolors.ENDC} to stop.")
    
    # --- IMPORTANT FOR JETSON NANO / LINUX L3 SNIFFING ---
    # Force Scapy to use L3RawSocket for sniffing. This operates at the IP layer
    # and can sometimes bypass issues with L2 (Ethernet) sniffing on certain devices.
    conf.L3socket = L3RawSocket
    print(f"{LOG_PREFIX} Using L3RawSocket for sniffing (IP layer).")
    
    # Optional: Still attempt to force pcap usage, though L3RawSocket might override some L2 pcap behavior
    conf.use_pcap = True
    print(f"{LOG_PREFIX} Attempting to force pcap usage for Scapy (secondary).")

    manager_thread = Thread(target=flow_manager, daemon=True)
    manager_thread.start()
    try:
        # Sniffing with L3RawSocket. Filter can still be applied at IP level.
        sniff(
            iface=args.interface,
            prn=packet_callback,
            store=False,
            filter="ip" # Filter for IP packets, as L3RawSocket handles IP layer and above
        )
    except (PermissionError, OSError):
         # Updated message for Linux/Ubuntu
         print(f"\n{bcolors.FAIL}{LOG_PREFIX} Permission denied. Please run this script with 'sudo'.{bcolors.ENDC}")
    except RuntimeError as e:
        if "not found" in str(e):
             # Updated message for Linux/Ubuntu
             print(f"\n{bcolors.FAIL}{LOG_PREFIX} Interface '{args.interface}' not found! Check 'ifconfig' or 'ip a' for available interfaces.{bcolors.ENDC}")
        else:
            print(f"\n{bcolors.FAIL}{LOG_PREFIX} A runtime error occurred: {e}{bcolors.ENDC}")
    except Exception as e:
        print(f"\n{bcolors.FAIL}{LOG_PREFIX} An unexpected error occurred: {e}{bcolors.ENDC}")
    finally:
        print(f"\n{bcolors.BOLD}--- Analyzer stopped ---{bcolors.ENDC}")
