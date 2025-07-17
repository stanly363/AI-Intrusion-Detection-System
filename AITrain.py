# --- Step 0: Import necessary libraries ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import opendatasets as od
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("TensorFlow Version:", tf.__version__)


# --- Step 1: Download and Load the Data ---
def download_and_load_data():
    """
    Downloads the NSL-KDD dataset from Kaggle if not present and loads it into pandas DataFrames.
    """
    dataset_url = 'https://www.kaggle.com/datasets/shivamalakkad/nslkdd-dataset'
    dataset_dir = 'nslkdd-dataset'
    train_file = os.path.join(dataset_dir, 'KDDTrain+.txt')
    test_file = os.path.join(dataset_dir, 'KDDTest+.txt')

    # Download the dataset if the directory doesn't exist
    if not os.path.isdir(dataset_dir):
        print(f"Downloading dataset from Kaggle: {dataset_url}")
        print("You will be prompted for your Kaggle username and API key.")
        od.download(dataset_url)
    else:
        print("Dataset directory already exists.")

    # Define column names for the dataset
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
                 "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
                 "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", 
                 "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
                 "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
                 "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
                 "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
                 "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", 
                 "attack_type", "difficulty_level"]
    
    # Load data from the files
    try:
        df_train = pd.read_csv(train_file, header=None, names=col_names)
        df_test = pd.read_csv(test_file, header=None, names=col_names)
        print("NSL-KDD datasets loaded successfully.")
        return df_train, df_test
    except FileNotFoundError:
        print(f"Error: Could not find dataset files at {train_file} and {test_file}.")
        print("Please ensure the download was successful and the paths are correct.")
        return None, None

# Execute data loading
df_train, df_test = download_and_load_data()

if df_train is None or df_test is None:
    exit()

# --- Step 2: Preprocess the Data ---
print("\n--- Preprocessing Data ---")

# Drop the 'difficulty_level' column as it's not needed for classification
df_train = df_train.drop('difficulty_level', axis=1)
df_test = df_test.drop('difficulty_level', axis=1)

# Convert the 'attack_type' text labels into binary numerical labels: 0 for 'normal', 1 for 'attack'
df_train['attack_binary'] = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['attack_binary'] = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

# Separate features (X) and the binary target label (y)
X_train_raw = df_train.drop(['attack_type', 'attack_binary'], axis=1)
y_train = df_train['attack_binary']
X_test_raw = df_test.drop(['attack_type', 'attack_binary'], axis=1)
y_test = df_test['attack_binary']

# Identify which columns are text-based (categorical) and which are numerical
categorical_cols = X_train_raw.select_dtypes(include=['object']).columns
numerical_cols = X_train_raw.select_dtypes(include=np.number).columns

# Use One-Hot Encoding on categorical features
X_train = pd.get_dummies(X_train_raw, columns=categorical_cols, dummy_na=False)
X_test = pd.get_dummies(X_test_raw, columns=categorical_cols, dummy_na=False)

# Align columns between training and test sets to ensure they have the same features
train_cols = X_train.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_test = X_test[X_train.columns]

# Scale numerical features to a standard range
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Data preprocessed. Training features shape: {X_train.shape}")


# --- Step 3: Build the AI Model ---
print("\n--- Building the AI Model ---")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.summary()


# --- Step 4: Train the AI Model ---
print("\n--- Training the AI Model ---")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.15,
                    callbacks=[early_stopping],
                    verbose=1)
print("Model training finished.")


# --- Step 5: Evaluate the Model's Performance ---
print("\n--- Evaluating Model Performance ---")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Create, save, and show the confusion matrix plot
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Attack'], yticklabels=['Actual Normal', 'Actual Attack'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Step 6: Save the Trained Model ---
print("\n--- Saving the Trained Model ---")
model.save('ids_model.h5')
print("Model saved as 'ids_model.h5'")
