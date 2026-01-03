# 📦 Black Box Audio Protocol (v3.0 Batch Edition)

A sovereign, serverless architecture for processing long-form daily audio logs into searchable text and identifying speakers using unsupervised learning.

## 📂 Project Index

### Configuration
* `config.py`: Central configuration loading paths and settings from environment variables.
* `.env.example`: Template for API keys and local paths.

### Core Logic
* `main.py`: The daily batch runner. Execute this script to process new files.
* `pipeline.py`: The processing engine. Handles Transcription -> Alignment -> Diarization -> Identification for a single file.
* `clustering.py`: Maintenance script to find and merge duplicate speaker identities.

### Support Modules
* `models.py`: AI Model Manager. Loads WhisperX and Pyannote into GPU memory efficiently.
* `utils.py`: Helper functions for file management, logging, and timestamp parsing.

### Dependencies
* `requirements.txt`: Python packages required to run the project.

## 🚀 Quick Start (Google Colab)

1.  **Setup:** Clone this repository (or copy these files) into your Google Drive project folder (e.g., `My Drive/yuzhe`).
2.  **Install:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure:** Rename `.env.example` to `.env` and add your Hugging Face Token.
4.  **Run:**
    ```bash
    python main.py
    ```
