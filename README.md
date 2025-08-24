# AI Intrusion Detection System 🚀

## Overview

This project provides an advanced AI-based Intrusion Detection System (IDS) built with Python, TensorFlow, and Scapy. It can be trained on the included dataset of network traffic captures to learn the difference between normal and malicious network traffic, and then used to analyze live network connections in real-time.

The system is composed of two main scripts:
* `AITrain.py`: A script to preprocess the included dataset, perform feature selection, and train a hyperparameter-tuned neural network.
* `IDS.py`: A script that uses a trained model to monitor a live network interface and alert on suspicious activity.

---

## Project Structure

The project uses the following directory structure. The `pretrained/` folder contains a ready-to-use model and its associated files.

```
/ai-intrusion-detection-system
|-- /pcap_samples
|   |-- /benign
|   |-- /malicious
|-- /pretrained
|   |-- ids_final_model.keras
|   |-- scaler.gz
|   |-- model_columns.pkl
|   |-- best_threshold.pkl
|-- AITrain.py
|-- IDS.py
|-- README.md
```

---

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### 1. Prerequisites

First, clone the repository and install the necessary software and libraries.

**For all Operating Systems:**
```bash
# Clone the repository
git clone https://github.com/stanly363/ai-intrusion-detection-system.git
cd ai-intrusion-detection-system

# Install the required Python libraries
pip install tensorflow pandas scapy numpy scikit-learn matplotlib seaborn joblib
```

**For Windows Users (Required):**
You must install **Npcap** for packet capturing to work.
1.  Download the latest installer from the [**Npcap Official Website**](https://npcap.com).
2.  During installation, ensure the following options are checked:
    * **"Support raw 802.11 traffic (and monitor mode)"**
    * **"Install Npcap in WinPcap API-compatible Mode"**

### 2. Training a New Model (Optional)

While a high-performance pre-trained model is provided, you can also train your own model from scratch by following these steps.

1.  **Run the Training Script:**
    Execute the `AITrain.py` script from your terminal. This will create a new set of four model files in the project's root directory.
    ```bash
    python AITrain.py
    ```
2. **Add Your Own Benign Traffic (Recommended for higher accuracy):**
    For the best performance, it is highly recommended to add your own benign network traffic to the training data. This will help the model learn the patterns of your specific network environment. You can do this by:
    * Capturing your own network traffic using a tool like Wireshark and saving it as a `.pcap` file.
    * Adding the captured `.pcap` file to the `/pcap_samples/benign/` directory.
    * Re-running the training script.

---

## Usage

The script needs root/administrator privileges and a network interface to monitor. You can choose to use the included pre-trained model or a model you have trained yourself.

### Using the Pre-Trained Model (Recommended)
To use the high-performance model included in the `pretrained/` folder, add the `--use-pretrained` flag to the command.

**On Linux/macOS:**
```bash
# Example using a wired interface
sudo python IDS.py --interface eth0 --use-pretrained
```

**On Windows (as Administrator):**
```powershell
# Example using a Wi-Fi interface
python IDS.py --interface "Wi-Fi" --use-pretrained
```

### Using a Self-Trained Model
If you have run `AITrain.py` and have your own set of model files in the project's root directory, simply run the command **without** the `--use-pretrained` flag.

**On Linux/macOS:**
```bash
sudo python IDS.py --interface eth0
```

**On Windows (as Administrator):**
```powershell
python IDS.py --interface "Wi-Fi"
```

---

## Troubleshooting

**Error: `Layer ['tcp'] not found` or `Interface not found` on Windows**

This is a common issue related to how packet capture libraries interact with Windows drivers. If problems persist:
1.  **Reinstall Npcap:** Ensure you have installed Npcap with the correct settings mentioned in the prerequisites.
2.  **Check Interface Name:** Use `python -c "from scapy.all import get_windows_if_list; print(get_windows_if_list())"` to find the exact name.
3.  **Wi-Fi Adapter Limitations:** Most built-in laptop Wi-Fi cards do not support monitor mode on Windows. If the script fails on Wi-Fi but works on a wired Ethernet connection, this is the cause.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
