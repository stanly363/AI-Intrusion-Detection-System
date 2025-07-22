
# --- Step 0: Import necessary libraries ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
import keras_tuner as kt
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Ether, Raw # For PCAP processing
import datetime # For packet timestamps
from collections import defaultdict, deque # For micro-flow tracking
import sys # For sys.exit

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("TensorFlow Version:", tf.__version__)

# --- Micro-Flow Configuration ---
MICRO_FLOW_WINDOW_SEC = 2.0 # Time window for micro-flow aggregation

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

micro_flow_manager_trainer = MicroFlowManager()


# --- Feature Extraction Function (Packet-Level + Micro-Flow) ---
def extract_combined_features(packet, packet_timestamp):
    """
    Extracts basic packet features and augments them with micro-flow features.
    This function MUST BE IDENTICAL to the one used in the live_ids_compatible.py script.
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
    
    _ , micro_features = micro_flow_manager_trainer.update_and_get_features(packet, packet_timestamp)
    
    if micro_features is None: 
        micro_features = {
            'micro_flow_pkt_count': 0, 'micro_flow_byte_count': 0,
            'micro_flow_unique_dst_ports': 0, 'micro_flow_syn_count': 0,
            'micro_flow_duration': 0.0, 'micro_flow_rate': 0.0
        }
    
    combined_features = {**packet_features, **micro_features}
    return combined_features


# --- Step 1: Load Data from Local PCAP Files ---
def load_data_from_local_pcaps(pcap_directory='pcap_samples'):
    """
    Loads data from local PCAP files, assuming a 'benign' and 'malicious' subdirectory
    or specific naming conventions.
    """
    print(f"--- Loading data from local PCAP directory: {pcap_directory} ---")
    
    all_extracted_features = []
    all_labels = []

    pcap_paths = []
    
    # Define how to find and label your PCAPs
    for root, _, files in os.walk(pcap_directory):
        label = -1
        if "benign" in root.lower():
            label = 0
        elif "malicious" in root.lower() or "attack" in root.lower():
            label = 1
        
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                if label != -1:
                    full_pcap_path = os.path.join(root, file)
                    pcap_paths.append((full_pcap_path, label))
                else:
                    print(f"{bcolors.WARNING}Warning: PCAP '{full_pcap_path}' found but no label assigned based on directory name. Skipping.{bcolors.ENDC}")

    if not pcap_paths:
        print(f"{bcolors.FAIL}Error: No PCAP files found in '{pcap_directory}' or its subdirectories. Please place small PCAP files there and organize them into 'benign'/'malicious' subfolders, or modify the script to point to specific files.{bcolors.ENDC}")
        print("Recommended: Create 'pcap_samples/benign/' and 'pcap_samples/malicious/' folders and put your .pcap files in them.")
        return pd.DataFrame(), pd.Series()

    # Reset micro_flow_manager for a fresh start for this training run
    global micro_flow_manager_trainer
    micro_flow_manager_trainer = MicroFlowManager()

    for full_pcap_path, label in pcap_paths:
        print(f"Processing {full_pcap_path} (Assigned Label: {label})")
        try:
            packets = rdpcap(full_pcap_path) 
            for i, packet in enumerate(packets):
                # Use packet.time for realistic timestamps if available, else simulate for flow tracking
                packet_timestamp = float(packet.time) if hasattr(packet, 'time') else (i / 1000.0) 
                
                features = extract_combined_features(packet, packet_timestamp)
                all_extracted_features.append(features)
                all_labels.append(label)
            print(f"  -> Extracted {len(packets)} packets.")
        except Exception as e:
            print(f"{bcolors.WARNING}  Warning: Could not read PCAP {full_pcap_path}: {e}{bcolors.ENDC}")

    if not all_extracted_features:
        print(f"{bcolors.FAIL}Error: No PCAP files processed or no features extracted. Exiting.{bcolors.ENDC}")
        return pd.DataFrame(), pd.Series()

    df = pd.DataFrame(all_extracted_features)
    df = df.fillna(0) # Fill any NaN values with 0

    print(f"Total extracted {len(df)} packet feature sets.")
    return df, pd.Series(all_labels)

# Call the new data loading function
X_raw, y = load_data_from_local_pcaps()

if X_raw.empty:
    print(f"{bcolors.FAIL}No data loaded. Exiting.{bcolors.ENDC}")
    sys.exit(1)

# --- Step 2: Initial Data Split (before feature selection) ---
# Split X_raw *before* feature selection and scaling
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
print(f"Initial raw data split. X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")


# --- Step 3: Advanced Optimization - Feature Selection ---
print("\n--- Performing Feature Selection with Random Forest ---")
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Fit Random Forest on the raw training data
feature_selector.fit(X_train_raw, y_train)

importances = feature_selector.feature_importances_
# Ensure N_FEATURES doesn't exceed the actual number of features available
N_FEATURES = min(50, X_train_raw.shape[1]) 
indices = np.argsort(importances)[::-1]
# Get the top N feature names from the raw training data columns
top_n_features = X_train_raw.columns[indices[:N_FEATURES]]

print(f"Selected Top {N_FEATURES} features.")
print(f"Top Features: {list(top_n_features)}")


# --- Step 4: Scale the SELECTED Features ---
print("\n--- Scaling Selected Features ---")
scaler = StandardScaler()

# Apply feature selection to both raw train and test sets
X_train_selected_raw = X_train_raw[top_n_features]
X_test_selected_raw = X_test_raw[top_n_features]

# Fit the scaler ONLY on the selected training features
X_train_scaled = scaler.fit_transform(X_train_selected_raw)
# Transform the test set using the scaler fitted on the training set
X_test_scaled = scaler.transform(X_test_selected_raw)

# Convert scaled arrays back to DataFrames, preserving column names.
# This is crucial for KerasTuner and the final model's input shape/feature names.
X_train_selected = pd.DataFrame(X_train_scaled, columns=top_n_features)
X_test_selected = pd.DataFrame(X_test_scaled, columns=top_n_features)

# --- DEBUGGING START (for trainer) ---
print(f"\n{bcolors.OKBLUE}DEBUG (Trainer): Columns of X_train_selected (input to Keras model):{bcolors.ENDC}")
print(X_train_selected.columns.tolist())
print(f"Number of columns: {len(X_train_selected.columns)}")

print(f"\n{bcolors.OKBLUE}DEBUG (Trainer): `top_n_features` which will be saved as model_columns.pkl:{bcolors.ENDC}")
print(top_n_features.tolist())
print(f"Number of `top_n_features`: {len(top_n_features)}")
# --- DEBUGGING END ---

print(f"Selected features scaled. New training feature shape: {X_train_selected.shape}")


# --- Step 5: Hyperparameter Tuning on Selected Features ---
print("\n--- Building Model with KerasTuner on Selected Features---")

def build_model(hp):
    model = tf.keras.models.Sequential()
    
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units_1, activation='relu', input_shape=(X_train_selected.shape[1],)))
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout_1))

    if hp.Boolean('add_second_layer'):
        hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
        model.add(tf.keras.layers.Dense(units=hp_units_2, activation='relu'))
        hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)
        model.add(tf.keras.layers.Dropout(rate=hp_dropout_2))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='keras_tuner_live_compatible_ids', 
                     project_name='ids_live_compatible_optimized')

stop_early_tuner = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("\n--- Starting Hyperparameter Search on Selected Features ---")
tuner.search(X_train_selected, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early_tuner])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nBest hyperparameters found: {best_hps.values}")

final_model = tuner.hypermodel.build(best_hps)
final_model.summary()


# --- Step 6: Train the Final Model on Selected Features ---
print("\n--- Training the Final Model on Selected Features ---")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Using Class Weights: {class_weight_dict}")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = final_model.fit(X_train_selected, y_train,
                          epochs=100,
                          batch_size=128,
                          validation_split=0.15,
                          callbacks=[early_stopping, reduce_lr],
                          class_weight=class_weight_dict,
                          verbose=1)
print("Final model training finished.")


# --- Step 7: Evaluate and Tune Threshold ---
print("\n--- Evaluating Final Model Performance ---")
loss, accuracy = final_model.evaluate(X_test_selected, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred_probs = final_model.predict(X_test_selected, verbose=0).ravel() # Added verbose=0 to suppress output during evaluation

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"\nBest Threshold (for balanced F1-score): {best_threshold:.4f}")
print("This threshold balances precision and recall. You can adjust it based on your needs.")

y_pred_tuned = (y_pred_probs > best_threshold).astype(int)

print("\nClassification Report (at Optimal Threshold):")
print(classification_report(y_test, y_pred_tuned, target_names=['Normal', 'Attack']))

print("Confusion Matrix (at Optimal Threshold):")
cm = confusion_matrix(y_test, y_pred_tuned)
print(cm)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Attack'], yticklabels=['Actual Normal', 'Actual Attack'])
plt.title('Live-Compatible IDS Confusion Matrix (Packet+Micro-Flow)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('live_compatible_ids_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Step 8: Save the Final Model and Artifacts ---
print("\n--- Saving the Final Optimized Model and Artifacts ---")

final_model.save('ids_live_compatible_model.keras')
print("Model saved as 'ids_live_compatible_model.keras'")

joblib.dump(scaler, 'scaler_live_compatible.gz')
print("Scaler saved as 'scaler_live_compatible.gz'")

# IMPORTANT: Save top_n_features (as a list) for the live script to know which columns to use
joblib.dump(top_n_features.tolist(), 'model_columns_live_compatible.pkl')
print("Model columns (selected features) saved as 'model_columns_live_compatible.pkl'")

joblib.dump(best_threshold, 'best_threshold_live_compatible.pkl')
print(f"Optimal prediction threshold saved as 'best_threshold_live_compatible.pkl'")

# Save ALL possible feature names that `extract_combined_features` can produce.
# This is useful for the live script to create a full template DataFrame before selecting
# the `model_columns`.
joblib.dump(X_raw.columns.tolist(), 'all_extracted_features_for_live.pkl')
print("All possible extracted feature names for live script saved as 'all_extracted_features_for_live.pkl'")

print(f"\n{bcolors.OKGREEN}All artifacts for Live-Compatible IDS have been saved successfully!{bcolors.ENDC}")
