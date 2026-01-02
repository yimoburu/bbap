# Black Box Audio Protocol

A modular Python system for running in Google Colab to transcribe and identify speakers in audio files from Google Drive using WhisperX and Pyannote.

## Features
- **Automated Transcription**: Uses WhisperX (large-v3) for high-accuracy multilingual transcription.
- **Speaker Identifiction**: "Black Box" identity system tracks speakers across files using vector embeddings.
- **Privacy First**: Stores only mathematical vectors, no audio clips for identity tracking.
- **Google Drive Integration**: Automatically monitors a folder for new recordings.

## Installation & Usage in Google Colab

1. **Upload Code**: Upload the `bbap` folder (containing this README and all `.py` files) to your Google Drive (e.g., in `My Drive/bbap`).

2. **Open Colab**: Create a new Google Colab notebook.

3. **Set Runtime**: Go to `Runtime` > `Change runtime type` and select **T4 GPU** (or better).

4. **Secrets**:
   - Add your Hugging Face Token as a secret named `HF_TOKEN` in Colab (key icon on the left).
   - Ensure you have accepted terms for `pyannote/embedding` and `pyannote/speaker-diarization-3.1` (or relevant models) on Hugging Face.

5. **Run the System**:
   Copy and paste the following code into a Colab cell:

   ```python
   # 1. Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # 2. Install Dependencies
   import os
   # Assuming you uploaded code to 'My Drive/bbap'
   project_path = "/content/drive/My Drive/bbap" 
   os.chdir(project_path)
   
   !pip install -r requirements.txt
   
   # Fix for some colab envs if needed (restart runtime might be required after install)
   # !pip install -U --no-deps git+https://github.com/m-bain/whisperx.git

   # 3. Authenticate with Hugging Face (retrieved from Colab secrets)
   from google.colab import userdata
   os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
   
   # 4. Run Main
   !python main.py
   ```

## Configuration
Edit `config.py` to change:
- `INPUT_DIR`: Folder to watch for audio files.
- `PROJECT_ROOT`: Folder for output (transcripts, vectors).
- `CHECK_INTERVAL`: How often to check for new files.

## Output
- Transcripts: `.../Processed_Conversations/[filename].txt`
- Logs: `processed_log.txt`

## Local Mode (Mac/PC)

You can run this pipeline on your local machine without Google Drive.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # You might need specific torch installation for your OS (e.g., Mac Metal MPS)
   ```

2. **Run in Local Mode**:
   Set the `BBAP_LOCAL_MODE` environment variable to `1`.
   
   ```bash
   export BBAP_LOCAL_MODE=1
   export HF_TOKEN="your_hugging_face_token"
   python main.py
   ```
   
   - **Input Directory**: The script will look for audio files in `./local_input`.
   - **Output Directory**: Results will be saved to `./local_output`.
   - **Device**: Automatically selects `mps` (Mac), `cuda`, or `cpu`.
