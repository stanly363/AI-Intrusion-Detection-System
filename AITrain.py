# --- Step 0: Import necessary libraries ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import warnings
import joblib
import keras_tuner as kt # Import KerasTuner

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
print("TensorFlow Version:", tf.__version__)


# --- Step 1: Download and Load the Data ---
def download_and_load_data():
    """
    Downloads the UNSW-NB15 dataset using KaggleHub and loads it into pandas DataFrames.
    """
    print("--- Downloading and Loading Data ---")
    try:
        dataset_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        print(f"Dataset downloaded to: {dataset_path}")
        train_file_path = os.path.join(dataset_path, 'UNSW_NB15_training-set.csv')
        test_file_path = os.path.join(dataset_path, 'UNSW_NB15_testing-set.csv')
        df_train = pd.read_csv(train_file_path)
        df_test = pd.read_csv(test_file_path)
        print("UNSW-NB15 datasets loaded successfully.")
        return df_train, df_test
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None

df_train, df_test = download_and_load_data()
if df_train is None or df_test is None:
    exit()


# --- Step 2: Preprocess the Data ---
print("\n--- Preprocessing Data ---")
df_train = df_train.drop(['id', 'attack_cat'], axis=1)
df_test = df_test.drop(['id', 'attack_cat'], axis=1)

X_train_raw = df_train.drop('label', axis=1)
y_train = df_train['label']
X_test_raw = df_test.drop('label', axis=1)
y_test = df_test['label']

categorical_cols = X_train_raw.select_dtypes(include=['object']).columns
numerical_cols = X_train_raw.select_dtypes(include=np.number).columns

X_train = pd.get_dummies(X_train_raw, columns=categorical_cols, dummy_na=False)
X_test = pd.get_dummies(X_test_raw, columns=categorical_cols, dummy_na=False)

train_cols = X_train.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_test = X_test[X_train.columns]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print(f"Data preprocessed. Training features shape: {X_train.shape}")

# --- Optimization: Calculate Class Weights for Imbalanced Data ---
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Calculated Class Weights: {class_weight_dict}")


# --- Step 3: Hyperparameter Tuning with KerasTuner ---
print("\n--- Building Model with KerasTuner ---")

def build_model(hp):
    """Builds a tunable model for KerasTuner."""
    model = tf.keras.models.Sequential()
    
    # Tune the number of units in the first Dense layer
    hp_units_1 = hp.Int('units_1', min_value=64, max_value=256, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units_1, activation='relu', input_shape=(X_train.shape[1],)))
    
    # Tune the dropout rate for the first layer
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout_1))

    # Tune the number of units in the second Dense layer
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units_2, activation='relu'))

    # Tune the dropout rate for the second layer
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout_2))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate the tuner
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=20, # More epochs for a more thorough search
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='ids_hyperparameter_tuning')

# Create a callback to stop training early if the validation loss is not improving
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("\n--- Starting Hyperparameter Search ---")
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nBest hyperparameters found: {best_hps.values}")


# --- Step 4: Train the Final, Optimized AI Model ---
print("\n--- Training the Final Optimized AI Model ---")

# Build the model with the optimal hyperparameters
final_model = tuner.hypermodel.build(best_hps)
final_model.summary()

# Define callbacks for final training
# Optimization: Reduce learning rate when a metric has stopped improving.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = final_model.fit(X_train, y_train,
                          epochs=100, # Train for more epochs with early stopping
                          batch_size=128,
                          validation_split=0.15,
                          callbacks=[early_stopping, reduce_lr],
                          class_weight=class_weight_dict, # Apply class weights
                          verbose=1)
print("Final model training finished.")


# --- Step 5: Evaluate the Optimized Model's Performance ---
print("\n--- Evaluating Optimized Model Performance ---")

loss, accuracy = final_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred_probs = final_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Attack'], yticklabels=['Actual Normal', 'Actual Attack'])
plt.title('Optimized Model Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('optimized_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Step 6: Save the Trained Model and Supporting Files ---
print("\n--- Saving the Optimized Model ---")
final_model.save('ids_optimized_model.keras') # Save in the modern .keras format
print("Model saved as 'ids_optimized_model.keras'")

joblib.dump(scaler, 'scaler.gz')
joblib.dump(X_train.columns, 'model_columns.pkl')
print("Scaler and model columns saved.")
