# AI Intrusion Detection System 🚀

## Overview

This project provides a simple yet powerful AI-based Intrusion Detection System (IDS) built with Python, TensorFlow, and Scapy. It can be trained on the NSL-KDD dataset to learn the difference between normal and malicious network traffic, and then used to analyze live network connections in real-time.

The system is composed of two main scripts:
* `AITrain.py`: A script to automatically download the dataset, preprocess it, and train a neural network for intrusion detection.
* `IDS.py`: A script that uses the trained model to monitor a live network interface and alert on suspicious activity.

---

## Features ✨

* **Trainable Model:** Automatically downloads the NSL-KDD dataset from Kaggle and trains a deep learning model.
* **Live Analysis:** Captures live network traffic using Scapy from any network interface.
* **Real-time Alerts:** Reconstructs network flows on the fly and provides instant console alerts for detected malicious activity.
* **Pre-trained Option:** Easily use a pre-trained model to start analyzing traffic immediately.

---

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### 1. Prerequisites

First, clone the repository and install the necessary Python libraries.

```bash
# Clone the repository
git clone [https://github.com/your-username/ai-intrusion-detection-system.git](https://github.com/your-username/ai-intrusion-detection-system.git)
cd ai-intrusion-detection-system

# Install the required libraries
pip install tensorflow pandas scapy numpy scikit-learn matplotlib seaborn opendatasets joblib
```

### 2. Choose Your Path

You have two options: use the pre-trained model for quick setup or train a new model from scratch.

#### Option A: Use the Pre-trained Model (Recommended for Quick Use)

This is the fastest way to get started.

1.  **Download the Model Files:**
    Go to the **Releases** page of this GitHub repository. Download the following three files from the latest release:
    * `ids_model.h5`
    * `scaler.gz`
    * `model_columns.pkl`

2.  **Place the Files:**
    Move these three files into the root directory of the project you just cloned.

3.  **You're Ready!**
    You can now skip to the [**Usage**](#usage) section to start the live analysis.

***

#### Option B: Train Your Own Model

Follow these steps if you want to train the model from scratch.

1.  **Kaggle API Credentials:**
    The training script needs Kaggle credentials to download the dataset.
    * Go to your Kaggle account settings (`https://www.kaggle.com/settings`).
    * Click `Create New Token` to download a `kaggle.json` file.
    * When you run the training script for the first time, you will be prompted to enter your Kaggle username and the API key from your `kaggle.json` file.

2.  **Run the Training Script:**
    Execute the `AITrain.py` script from your terminal:
    ```bash
    python AITrain.py
    ```
    The script will handle everything: downloading the data, preprocessing it, training the model, and saving the three required artifact files (`ids_model.h5`, `scaler.gz`, and `model_columns.pkl`) in the project directory.

---

## Usage

Once you have the three model artifact files (either by downloading or by training), you can run the live IDS.

1.  **Configure the Network Interface:**
    Open the `IDS.py` file in a text editor. Find the following line and change `"eth0"` to the name of the network interface you want to monitor (e.g., `enp0s3`, `wlan0`). This should be the interface connected to your network's mirror or SPAN port.
    ```python
    # file: IDS.py
    NETWORK_INTERFACE = "eth0" 
    ```

2.  **Run the Live Analyzer:**
    The script needs root privileges to capture network packets. Run it using `sudo`:
    ```bash
    sudo python IDS.py
    ```

The system will now start monitoring traffic. It will print a real-time, color-coded alert to your console whenever it detects a connection that it classifies as malicious. Press `Ctrl+C` to stop the analyzer.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
