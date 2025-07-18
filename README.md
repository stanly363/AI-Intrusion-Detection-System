# AI Intrusion Detection System 🚀

## Overview

This project provides an advanced AI-based Intrusion Detection System (IDS) built with Python, TensorFlow, and Scapy. It can be trained on the **UNSW-NB15 dataset** to learn the difference between normal and malicious network traffic, and then used to analyze live network connections in real-time.

The system is composed of two main scripts:
* `AITrain.py`: A script to automatically download the dataset, preprocess it, perform feature selection, and train a hyperparameter-tuned neural network.
* `IDS.py`: A script that uses the trained model to monitor a live network interface and alert on suspicious activity.

---

## Features ✨

* **Advanced Trainable Model:** Automatically downloads the UNSW-NB15 dataset and trains a deep learning model using sophisticated techniques like feature selection and hyperparameter tuning.
* **Optimized Predictions:** Uses a dynamically calculated prediction threshold to balance precision and recall, leading to more accurate alerts.
* **Live Analysis:** Captures live network traffic using Scapy from any network interface.
* **Real-time Alerts:** Reconstructs network flows on the fly and provides instant, color-coded console alerts for detected malicious activity.
* **Cross-Platform:** Includes specific configurations and instructions for running on Windows, macOS, and Linux.
* **Pre-trained Option:** Easily use a pre-trained model to start analyzing traffic immediately.

---

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### 1. Prerequisites

First, clone the repository and install the necessary software and libraries.

**For all Operating Systems:**
```bash
# Clone the repository
git clone [https://github.com/your-username/ai-intrusion-detection-system.git](https://github.com/your-username/ai-intrusion-detection-system.git)
cd ai-intrusion-detection-system

# Install the required Python libraries
pip install tensorflow pandas scapy numpy scikit-learn matplotlib seaborn joblib kagglehub keras-tuner
```

**For Windows Users (Required):**
You must install **Npcap** for packet capturing to work.
1.  Download the latest installer from the [**Npcap Official Website**](https://npcap.com).
2.  During installation, ensure the following options are checked:
    * **"Support raw 802.11 traffic (and monitor mode)"**
    * **"Install Npcap in WinPcap API-compatible Mode"**

### 2. Choose Your Path

You have two options: use the pre-trained model for a quick setup or train a new model from scratch.

#### Option A: Use the Pre-trained Model (Recommended)

This is the fastest way to get started.

1.  **Verify the Model Files:**
    Ensure the following **four** files are present in the project's root directory:
    * `ids_final_model.keras`
    * `scaler.gz`
    * `model_columns.pkl`
    * `best_threshold.pkl`

2.  **You're Ready!**
    You can now skip to the [**Usage**](#usage) section to start the live analysis.

***

#### Option B: Train Your Own Model

Follow these steps if you want to train the model from scratch.

1.  **Kaggle API Credentials:**
    The training script needs Kaggle credentials to download the dataset.
    * Go to your Kaggle account settings (`https://www.kaggle.com/settings`).
    * Click `Create New Token` to download a `kaggle.json` file.
    * The script will prompt for your Kaggle username and API key from this file when you first run it.

2.  **Run the Training Script:**
    Execute the `AITrain.py` script from your terminal:
    ```bash
    python AITrain.py
    ```
    The script will handle everything: downloading data, preprocessing, training the advanced model, and saving the four required artifact files (`ids_final_model.keras`, `scaler.gz`, `model_columns.pkl`, and `best_threshold.pkl`) in the project directory.

---

## Usage

The script needs root/administrator privileges to capture network packets and requires you to specify which network interface to monitor.

### For Linux & macOS

1.  **Find your interface name** using commands like `ifconfig` or `ip a`. Common names are `eth0` (wired) or `en0`, `wlan0` (wireless).

2.  **Run the script** from your terminal using `sudo`:
    ```bash
    # Example for a wired interface
    sudo python IDS.py --interface eth0

    # Example for a wireless interface
    sudo python IDS.py --interface wlan0
    ```

### For Windows

1.  **Find your interface name**. The easiest way is to run the following command in a terminal, which will list the exact names `scapy` can use:
    ```powershell
    python -c "from scapy.all import get_windows_if_list; print(get_windows_if_list())"
    ```
    Look for the `name` field of your "Wi-Fi" or "Ethernet" adapter in the output.

2.  **Run the script** from a **Command Prompt or PowerShell opened as Administrator**:
    ```powershell
    # Example for a Wi-Fi interface (use quotes if the name has spaces)
    python IDS.py --interface "Wi-Fi"

    # Example for an Ethernet interface
    python IDS.py --interface "Ethernet"
    ```

The system will now start monitoring traffic and will print a real-time, color-coded alert to your console whenever it detects a malicious connection. Press `Ctrl+C` to stop the analyzer.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
