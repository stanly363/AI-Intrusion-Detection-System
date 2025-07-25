# live_ids_compatible.py
# --- Live IDS: Step 0: Import necessary libraries ---
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw # For live packet capture
import time # For current timestamp
import datetime # For display purposes
import warnings
import sys
import os # Added for path manipulation
import argparse # Added for command-line argument parsing
from collections import defaultdict, deque # For micro-flow tracking

# Suppress scapy warnings (e.g., No route found)
warnings.filterwarnings('ignore', category=UserWarning, module='scapy')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='scapy')

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


# --- Comprehensive IP Allowlist Configuration ---
ALLOWLIST_IPS = {
    # --- 1. Essential Local Network ---
    '192.168.10.1',      # Common router/gateway IP

    # --- 2. Public DNS Servers ---
    '8.8.8.8',          # Google Public DNS
    '8.8.4.4',          # Google Public DNS
    '1.1.1.1',          # Cloudflare Public DNS
    '1.0.0.1',          # Cloudflare Public DNS

    # Apple (macOS/iOS Updates, iCloud, App Store, etc.)
    '17.0.0.0/8',

    # Microsoft (Windows Update, Office 365, Azure, etc.)
    '13.107.0.0/16',
    '40.74.0.0/15',
    '52.96.0.0/14',
    '74.125.0.0/16',    # Google Services (Search, Gmail, etc.)
    '172.217.0.0/16',   # Google Services
    '34.0.0.0/8',       # Google Cloud Platform
    '35.184.0.0/13',    # Google Cloud Platform
    '104.16.0.0/12',    # Cloudflare
    '172.64.0.0/13',   # Cloudflare
    '151.101.0.0/16',  # Fastly
    '23.0.0.0/11',     # Akamai
    '104.64.0.0/10'    # Akamai
}

# --- Micro-Flow Configuration (MUST MATCH TRAINER'S) ---
MICRO_FLOW_WINDOW_SEC = 2.0 # Time window for micro-flow aggregation

# MicroFlowInfo and MicroFlowManager MUST BE IDENTICAL TO TRAINER'S
class MicroFlowInfo:
    """Stores information for a single micro-flow."""
    def __init__(self, start_time):
        self.start_time = start_time
        self.last_packet_time = start_time
        self.packet_data = deque([(start_time, 0, 0, False)]) # (timestamp, size, dst_port, is_syn)
        
    def update(self, current_time, dst_port, packet_size, is_syn):
        self.last_packet_time = current_time
        self.packet_data.append((current_time, packet_size, dst_port, is_syn))
        while self.packet_data and (current_time - self.packet_data[0][0]) > MICRO_FLOW_WINDOW_SEC:
            self.packet_data.popleft()
        
        self.pkt_count = len(self.packet_data)
        self.byte_count = sum(data[1] for data in self.packet_data)
        self.unique_dst_ports = {data[2] for data in self.packet_data}
        self.syn_count = sum(1 for data in self.packet_data if data[3])
    
    def get_features(self):
        if self.packet_data:
            micro_flow_duration_actual = self.packet_data[-1][0] - self.packet_data[0][0]
            if micro_flow_duration_actual == 0: micro_flow_duration_actual = 1e-6
        else:
            micro_flow_duration_actual = 1e-6

        return {
            'micro_flow_pkt_count': self.pkt_count,
            'micro_flow_byte_count': self.byte_count,
            'micro_flow_unique_dst_ports': len(self.unique_dst_ports),
            'micro_flow_syn_count': self.syn_count,
            'micro_flow_duration': micro_flow_duration_actual,
            'micro_flow_rate': self.pkt_count / micro_flow_duration_actual if micro_flow_duration_actual > 0 else 0
        }

class MicroFlowManager:
    """Manages multiple active micro-flows."""
    def __init__(self, window_size=MICRO_FLOW_WINDOW_SEC):
        self.flows = defaultdict(lambda: None) 
        self.window_size = window_size

    def _get_flow_key(self, src_ip, dst_ip, proto_num):
        if src_ip < dst_ip:
            return (src_ip, dst_ip, proto_num)
        else:
            return (dst_ip, src_ip, proto_num)

    def update_and_get_features(self, packet, packet_timestamp):
        if not IP in packet:
            return None, None

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto_num = packet[IP].proto
        dst_port = 0
        is_syn = False

        if TCP in packet:
            dst_port = packet[TCP].dport
            if packet[TCP].flags and 'S' in str(packet[TCP].flags):
                is_syn = True
        elif UDP in packet:
            dst_port = packet[UDP].dport

        flow_key = self._get_flow_key(src_ip, dst_ip, proto_num)
        packet_size = len(packet)

        if self.flows[flow_key] is None:
            self.flows[flow_key] = MicroFlowInfo(packet_timestamp)
        
        self.flows[flow_key].update(packet_timestamp, dst_port, packet_size, is_syn)
        
        flows_to_prune = []
        for key, flow_info in list(self.flows.items()):
            if flow_info and (packet_timestamp - flow_info.last_packet_time) > self.window_size * 2:
                flows_to_prune.append(key)
        for key in flows_to_prune:
            del self.flows[key]

        return flow_key, self.flows[flow_key].get_features()

# Instantiate the MicroFlowManager for live processing
micro_flow_manager_live = MicroFlowManager()


# --- Feature Extraction Function (Packet-Level + Micro-Flow) ---
# THIS FUNCTION MUST BE IDENTICAL TO THE ONE IN THE TRAINER!
def extract_combined_features(packet, packet_timestamp):
    """
    Extracts basic packet features and augments them with micro-flow features.
    This function MUST EXACTLY MATCH the one used in the trainer script.
    """
    packet_features = {}
    packet_features['ip_len'] = 0
    packet_features['ip_ttl'] = 0
    packet_features['ip_id'] = 0
    packet_features['ip_flags_DF'] = 0
    packet_features['ip_flags_MF'] = 0
    packet_features['tcp_sport'] = 0
    packet_features['tcp_dport'] = 0
    packet_features['tcp_flags_SYN'] = 0
    packet_features['tcp_flags_ACK'] = 0
    packet_features['tcp_flags_FIN'] = 0
    packet_features['tcp_flags_RST'] = 0
    packet_features['tcp_flags_PSH'] = 0
    packet_features['tcp_flags_URG'] = 0
    packet_features['tcp_window'] = 0
    packet_features['udp_sport'] = 0
    packet_features['udp_dport'] = 0
    packet_features['udp_len'] = 0
    packet_features['icmp_type'] = 0
    packet_features['icmp_code'] = 0
    packet_features['packet_size'] = len(packet)
    packet_features['payload_len'] = 0
    packet_features['proto_num'] = 0

    if IP in packet:
        packet_features['ip_len'] = packet[IP].len
        packet_features['ip_ttl'] = packet[IP].ttl
        packet_features['ip_id'] = packet[IP].id
        if packet[IP].flags:
            if 'DF' in str(packet[IP].flags): packet_features['ip_flags_DF'] = 1
            if 'MF' in str(packet[IP].flags): packet_features['ip_flags_MF'] = 1
        
        packet_features['proto_num'] = packet[IP].proto

    # Ensure to check if TCP/UDP/ICMP layers exist before trying to access their fields
    if TCP in packet:
        packet_features['tcp_sport'] = packet[TCP].sport
        packet_features['tcp_dport'] = packet[TCP].dport
        flags = str(packet[TCP].flags)
        if 'S' in flags: packet_features['tcp_flags_SYN'] = 1
        if 'A' in flags: packet_features['tcp_flags_ACK'] = 1
        if 'F' in flags: packet_features['tcp_flags_FIN'] = 1
        if 'R' in flags: packet_features['tcp_flags_RST'] = 1
        if 'P' in flags: packet_features['tcp_flags_PSH'] = 1
        if 'U' in flags: packet_features['tcp_flags_URG'] = 1
        packet_features['tcp_window'] = packet[TCP].window
        if Raw in packet:
            packet_features['payload_len'] = len(packet[Raw].load)

    elif UDP in packet:
        packet_features['udp_sport'] = packet[UDP].sport
        packet_features['udp_dport'] = packet[UDP].dport
        packet_features['udp_len'] = packet[UDP].len
        if Raw in packet:
            packet_features['payload_len'] = len(packet[Raw].load)

    elif ICMP in packet:
        packet_features['icmp_type'] = packet[ICMP].type
        packet_features['icmp_code'] = packet[ICMP].code
        if Raw in packet:
            packet_features['payload_len'] = len(packet[Raw].load)
    
    _ , micro_features = micro_flow_manager_live.update_and_get_features(packet, packet_timestamp)
    
    if micro_features is None: 
        micro_features = {
            'micro_flow_pkt_count': 0, 'micro_flow_byte_count': 0,
            'micro_flow_unique_dst_ports': 0, 'micro_flow_syn_count': 0,
            'micro_flow_duration': 0.0, 'micro_flow_rate': 0.0
        }
    
    combined_features = {**packet_features, **micro_features}
    return combined_features


# Global variables to hold loaded models and artifacts
final_model = None
scaler = None
required_feature_names = None
best_threshold = None
last_attack_source_ip = None

# --- Live IDS: Step 1: Load the saved models and artifacts ---
def load_ids_artifacts(model_path_prefix=""):
    """Loads the IDS model and associated artifacts."""
    global final_model, scaler, required_feature_names, best_threshold 
    
    print("--- Loading IDS Artifacts ---")
    try:
        final_model = tf.keras.models.load_model(os.path.join(model_path_prefix, 'ids_live_compatible_model.keras'))
        scaler = joblib.load(os.path.join(model_path_prefix, 'scaler_live_compatible.gz'))
        model_columns = joblib.load(os.path.join(model_path_prefix, 'model_columns_live_compatible.pkl'))
        best_threshold_from_training = joblib.load(os.path.join(model_path_prefix, 'best_threshold_live_compatible.pkl'))
        
        best_threshold = 0.9
        
        required_feature_names = model_columns 

        print(f"{bcolors.OKGREEN}Models and artifacts loaded successfully from '{model_path_prefix}'!{bcolors.ENDC}")
        print(f"Optimal Prediction Threshold: {best_threshold:.4f} (Originally optimized to: {best_threshold_from_training:.4f})")

        print(f"\n{bcolors.OKBLUE}DEBUG (Live IDS): `required_feature_names` loaded from model_columns.pkl:{bcolors.ENDC}")
        print(required_feature_names)
        print(f"Number of required_feature_names: {len(required_feature_names)}")

    except Exception as e:
        print(f"{bcolors.FAIL}Error loading artifacts from '{model_path_prefix}': {e}{bcolors.ENDC}")
        print(f"Please ensure the necessary files are in the '{model_path_prefix}' directory.")
        sys.exit(1)


# --- Live IDS: Step 2: Define packet processing callback function ---
def process_packet(packet):
    global last_attack_source_ip

    current_time_unix = time.time()
    current_time_display = datetime.datetime.fromtimestamp(current_time_unix)
    
    if not IP in packet:
        return

    # Check against allowlist before processing
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    if src_ip in ALLOWLIST_IPS or dst_ip in ALLOWLIST_IPS:
        return # Silently ignore packets from/to allowlisted IPs
        
    try:
        raw_features = extract_combined_features(packet, current_time_unix)
        extracted_df = pd.DataFrame([raw_features])
        features_to_scale = pd.DataFrame(0.0, index=[0], columns=required_feature_names)
        
        for col in extracted_df.columns:
            if col in features_to_scale.columns:
                features_to_scale[col] = extracted_df[col]

        if features_to_scale.isnull().any().any():
            print(f"[{current_time_display.strftime('%Y-%m-%d %H:%M:%S')}] {bcolors.FAIL}DEBUG (Live IDS): NaN values detected in features_to_scale BEFORE scaling!{bcolors.ENDC}")

        features_df_scaled = scaler.transform(features_to_scale)
        prediction_prob = final_model.predict(features_df_scaled, verbose=0)[0][0] 
        prediction_label = "ATTACK" if prediction_prob > best_threshold else "NORMAL"

        if prediction_label == "ATTACK":
            # --- NEW: Manual rule for consecutive attacks ---
            # Check if the current attacker is the same as the last one
            if src_ip == last_attack_source_ip:
                # This is the second (or more) consecutive attack from this IP, so we print the alert.
                status_color = bcolors.FAIL
                protocol_name_display = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(packet[IP].proto, 'OTHER')

                print(f"[{current_time_display.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"[{status_color}{prediction_label}{bcolors.ENDC}] "
                      f"Packet: {src_ip} -> {dst_ip} (Proto: {protocol_name_display}) | "
                      f"Size: {len(packet)} bytes | "
                      f"Prob: {prediction_prob:.4f})")
                print(f"{bcolors.WARNING}!!! POTENTIAL INTRUSION DETECTED !!! (Packet from {src_ip}){bcolors.ENDC}")
            
            # Update the last attacker IP regardless of whether we printed.
            # This "arms" the system for the next packet.
            last_attack_source_ip = src_ip
            # --- END NEW ---
        else:
            # If the packet is normal, reset the last attacker tracker.
            last_attack_source_ip = None

    except Exception as e:
        src_ip_str = packet[IP].src if IP in packet else "N/A"
        dst_ip_str = packet[IP].dst if IP in packet else "N/A"
        print(f"[{current_time_display.strftime('%Y-%m-%d %H:%M:%S')}] {bcolors.FAIL}Error processing packet from {src_ip_str} to {dst_ip_str}: {e}{bcolors.ENDC}")

    finally:
        global micro_flow_manager_live
        flows_to_prune = []
        for key, flow_data in list(micro_flow_manager_live.flows.items()): 
            if flow_data and (current_time_unix - flow_data.last_packet_time) > MICRO_FLOW_WINDOW_SEC * 2: 
                flows_to_prune.append(key)
        for key in flows_to_prune:
            del micro_flow_manager_live.flows[key]

# --- Live IDS: Step 3: Start sniffing network traffic ---
def start_live_ids(interface=None):
    """Starts sniffing live network traffic and applies the IDS model."""
    print(f"\n--- Starting Live IDS Monitoring on interface: {interface if interface else 'all available interfaces'} ---")
    print("Press Ctrl+C to stop.")
    try:
        sniff(prn=process_packet, store=0, iface=interface, filter="ip") 
    except KeyboardInterrupt:
        print(f"\n{bcolors.OKCYAN}IDS monitoring stopped by user.{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}An error occurred during sniffing: {e}{bcolors.ENDC}")
        if "Permission denied" in str(e) or "You don't have permission" in str(e):
            print(f"{bcolors.WARNING}Hint: Try running with sudo/administrator privileges.{bcolors.ENDC}")
        elif "No such device" in str(e) or "No such interface" in str(e):
             print(f"{bcolors.WARNING}Hint: Check your network interface name.{bcolors.ENDC}")
        else:
            raise

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live Intrusion Detection System. Monitor network traffic for anomalies.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--interface', required=True,
                        help='Network interface to monitor (e.g., "WiFi", "Ethernet", "eth0", "en0").')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use models from the "pretrained" subdirectory.')
    
    args = parser.parse_args()

    model_load_path = ""
    if args.pretrained:
        model_load_path = "pretrained"
        if not os.path.exists(model_load_path):
            print(f"{bcolors.FAIL}Error: Pretrained models directory '{model_load_path}' not found.{bcolors.ENDC}")
            sys.exit(1)
    
    load_ids_artifacts(model_load_path)
    start_live_ids(interface=args.interface)
