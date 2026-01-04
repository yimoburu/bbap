# Black Box Audio Protocol (BBAP)

**Modular Audio Processing Pipeline for Transcription, Diarization, and Speaker Identity Tracking.**

This project uses **WhisperX** (transcription + alignment) and **Pyannote.Audio** (speaker diarization + embedding) to process long-form audio. It includes a self-learning identity system that tracks speakers across multiple files using vector database clustering.

---

## 🏗 System Architecture

The project is designed as a modular pipeline. Any agent or developer rebuilding this project should follow this structure:

### **Modules**
*   **`config.py`**: Central configuration. Handles path detection (Local vs Colab), device selection (CPU/CUDA/MPS), and environment variables.
*   **`models.py`**: Manages AI model loading.
    *   *Critical Fix*: Includes `torch.serialization.add_safe_globals` to fix PyTorch 2.6+ security errors.
    *   *Critical Fix*: Imports `whisperx.diarize` explicitly to avoid `AttributeError`.
*   **`pipeline.py`**: Core logic.
    *   Steps: Transcribe -> Align -> Diarize -> Identify.
    *   *Optimization*: Uses in-memory audio slicing (NumPy/Torch) to avoid `ffmpeg` format errors with `.m4a` files during the identity phase.
*   **`utils.py`**: Helper functions for file I/O, timestamp parsing, and ID management.
*   **`clustering.py`**: (Optional) For offline re-clustering of the vector database.
*   **`main.py`**: Entry point. iterators over the input folder and runs the pipeline.
    *   *Critical Fix*: Injects aggressive SSL certificate patching (`certifi`) to solve `[SSL: CERTIFICATE_VERIFY_FAILED]` errors.

### **Data Flow**
1.  **Input**: Audio files in `local_data/Voice Record Pro` (or Google Drive).
2.  **Process**:
    *   **WhisperX**: Generates text and timestamps.
    *   **Pyannote**: Separates speakers (Speaker_01, Speaker_02).
    *   **Identification**: Extracts voice fingerprints (embeddings) and compares them against a local Vector DB (`.npy` files).
3.  **Output**: Markdown transcriptions in `local_data/yuzhe/Processed_Conversations`.

---

## 🛠 Installation & Dependencies

### **Requirements**
See `requirements.txt`. Key dependencies:
*   `git+https://github.com/m-bain/whisperx.git` (Must be installed from Git)
*   `pyannote.audio`
*   `torch`, `torchaudio`
*   `python-dotenv`
*   `certifi` (for SSL fix)

### **Setup**

#### **Option A: Google Colab (Recommended)**
1.  Upload `bbap_colab.ipynb` to Google Colab.
2.  Run the cells. It will:
    *   Install all dependencies.
    *   Mount Google Drive.
    *   Clone this repository.
    *   Prompt for your Hugging Face Token.

#### **Option B: Local Machine (Mac/Linux/Windows)**
1.  **Clone Repository**:
    ```bash
    git clone https://github.com/yimoburu/bbap.git
    cd bbap
    ```
2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need `ffmpeg` installed on your system (`brew install ffmpeg`).*

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Hugging Face Token (Read permissions required)
# MUST accept user conditions for:
# 1. pyannote/embedding
# 2. pyannote/speaker-diarization-3.1
HF_TOKEN=your_token_here

# Whisper Model Size (tiny, base, small, medium, large-v3)
# 'medium' is recommended for local CPU/Mac dev.
WHISPER_MODEL=medium

# Language (Optional). Leave empty for auto-detect.
# Set to 'zh' ONLY if you want to force Chinese-only transcription.
WHISPER_LANGUAGE=

# Mode
LOCAL_MODE=True
```

---

## 🚀 Usage

Run the pipeline:
```bash
python main.py
```

The script will:
1.  Auto-detect new files in the input directory.
2.  Process them one by one.
3.  Generate a Markdown transcript with identified speakers.
4.  Update the Daily Briefing.

---

## 🔧 Troubleshooting (Known Issues)

### **1. SSL Certificate Verify Failed**
*   **Error**: `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]`
*   **Cause**: Python failing to find system certificates.
*   **Fix**: `main.py` automatically patches this using `certifi`. No manual action needed.

### **2. PyTorch 2.6 WeightsUnpickler Error**
*   **Error**: `WeightsUnpickler error: Unsupported global: GLOBAL omegaconf...`
*   **Cause**: PyTorch 2.6+ security restriction on `torch.load`.
*   **Fix**: `models.py` has a safe globals allowlist. Do not remove the `torch.serialization.add_safe_globals` block.

### **3. Hugging Face Authentication / Gated Models**
*   **Error**: `Could not download 'pyannote/speaker-diarization-3.1'`
*   **Fix**:
    1.  Ensure `HF_TOKEN` is set in `.env`.
    2.  **CRITICAL**: You must visit the Hugging Face model pages and **Accept User Conditions** for BOTH:
        *   [pyannote/embedding](https://hf.co/pyannote/embedding)
        *   [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)

### **4. "Format not recognised" (M4A/MP3)**
*   **Error**: Occurs during Identity Verification phase.
*   **Fix**: `pipeline.py` uses in-memory slicing of the loaded waveform instead of asking `torchaudio` to re-open the file. This supports all formats WhisperX supports (`m4a`, `mp3`, `wav`).

---
