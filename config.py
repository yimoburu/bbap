import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- PATH CONFIGURATION ---
# Local Mode Toggle
LOCAL_MODE = os.getenv('LOCAL_MODE', 'False').lower() == 'true'

if LOCAL_MODE:
    BASE_DIR = os.path.join(os.getcwd(), 'local_data')
    print(f"🔹 RUNNING IN LOCAL MODE. Data directory: {BASE_DIR}")
else:
    # Base Root of Google Drive
    BASE_DIR = os.getenv('BASE_DIR', '/content/drive/My Drive')

# Project Root Name (e.g., 'yuzhe')
PROJECT_ROOT_NAME = os.getenv('PROJECT_ROOT_NAME', 'yuzhe')
PROJECT_ROOT = os.path.join(BASE_DIR, PROJECT_ROOT_NAME)

# Input Folder Name (e.g., 'Voice Record Pro')
# This is where raw audio files from the app are saved
INPUT_FOLDER_NAME = os.getenv('INPUT_FOLDER_NAME', 'Voice Record Pro')
INPUT_FOLDER = os.path.join(BASE_DIR, INPUT_FOLDER_NAME)

# Output Paths (Nested in Project Root)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Processed_Conversations')
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'Voice_Bank/Vectors')
LOG_FILE = os.path.join(PROJECT_ROOT, 'processed_log.txt')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'Daily_Reports')

# --- PROCESSING SETTINGS ---
# Cosine Similarity Threshold (0.65 is standard for Pyannote)
# Higher = Stricter (Creates more "Person_XXX" identities)
# Lower = Looser (Might merge different people)
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.65'))

# If two "Unknown" vectors are this similar, the maintenance script suggests merging
CLUSTER_MERGE_THRESHOLD = float(os.getenv('CLUSTER_MERGE_THRESHOLD', '0.85'))

# --- AI PARAMETERS ---
import torch

def detect_config():
    if torch.cuda.is_available():
        return "cuda", "float16"
    elif torch.backends.mps.is_available():
        # WhisperX/CTranslate2 doesn't fully support MPS yet.
        # We default to CPU + int8 for stability on Mac.
        # You can try overriding DEVICE='mps' in .env if testing support.
        print("⚡️ MPS (Mac) detected. Defaulting to CPU (int8) for WhisperX compatibility.")
        return "cpu", "int8"
    else:
        return "cpu", "int8"

_default_device, _default_compute = detect_config()

# 'cuda' for GPU, 'cpu' for CPU, 'mps' for Mac M1/M2/M3
DEVICE = os.getenv('DEVICE', _default_device)
# 'float16' for GPU, 'float32' or 'int8' for CPU
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', _default_compute)
# Batch size for processing
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
# Whisper Model Size (tiny, base, small, medium, large-v3)
# Recommend 'medium' or 'small' for local CPU debugging
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3')
# Language Override (e.g. 'en', 'zh', or None to auto-detect)
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', None)
# Initial Prompt to guide the model (e.g. "Mixed English and Chinese")
WHISPER_INITIAL_PROMPT = os.getenv('WHISPER_INITIAL_PROMPT', None)

# Diarization speaker count (for conversations, typically 2)
# Set to None to auto-detect, or specify min/max for better accuracy
MIN_SPEAKERS = int(os.getenv('MIN_SPEAKERS', '2')) if os.getenv('MIN_SPEAKERS') else None
MAX_SPEAKERS = int(os.getenv('MAX_SPEAKERS', '10')) if os.getenv('MAX_SPEAKERS') else None

# Diarization model selection
# Options: "pyannote/speaker-diarization-3.1" (default, best quality)
#          "pyannote/speaker-diarization-3.0" (faster, slightly lower quality)
DIARIZATION_MODEL = os.getenv('DIARIZATION_MODEL', 'pyannote/speaker-diarization-3.1')

# Clustering threshold for diarization (lower = more speakers detected, higher = fewer)
# Range: 0.0 to 1.0, default: 0.7 (pyannote default)
# Lower values (0.5-0.6) = more sensitive, may create more speaker clusters
# Higher values (0.8-0.9) = less sensitive, may merge similar speakers
DIARIZATION_CLUSTERING_THRESHOLD = float(os.getenv('DIARIZATION_CLUSTERING_THRESHOLD', '0.7'))

# WhisperX VAD (Voice Activity Detection) parameters for better segmentation
# These are passed to whisperx.load_model() via vad_options
# Lower values = more sensitive (detects more segments, catches shorter speech)
# Higher values = less sensitive (fewer segments, only longer speech)
# Default WhisperX values: vad_onset=0.500, vad_offset=0.363, chunk_size=30
WHISPERX_VAD_METHOD = os.getenv('WHISPERX_VAD_METHOD', 'pyannote')  # Options: "pyannote" (default, more accurate) or "silero" (faster)
WHISPERX_VAD_ONSET = float(os.getenv('WHISPERX_VAD_ONSET', '0.5'))  # Speech start threshold (0.0-1.0)
WHISPERX_VAD_OFFSET = float(os.getenv('WHISPERX_VAD_OFFSET', '0.363'))  # Speech end threshold (0.0-1.0)
WHISPERX_VAD_CHUNK_SIZE = float(os.getenv('WHISPERX_VAD_CHUNK_SIZE', '5'))  # Max chunk size in seconds (default: 30). Smaller = more segments, less merging
WHISPERX_VAD_MIN_DURATION_ON = float(os.getenv('WHISPERX_VAD_MIN_DURATION_ON', '0.0'))  # Min speech duration (seconds)
WHISPERX_VAD_MIN_DURATION_OFF = float(os.getenv('WHISPERX_VAD_MIN_DURATION_OFF', '0.0'))  # Min silence to split (seconds)

# Pyannote VAD parameters (for external VAD if needed)
# Lower values = more sensitive (detects more segments, catches shorter speech)
# Higher values = less sensitive (fewer segments, only longer speech)
VAD_ONSET = float(os.getenv('VAD_ONSET', '0.3'))  # Speech start threshold (0.0-1.0)
VAD_OFFSET = float(os.getenv('VAD_OFFSET', '0.1'))  # Speech end threshold (0.0-1.0)
VAD_MIN_DURATION_ON = float(os.getenv('VAD_MIN_DURATION_ON', '0.0'))  # Min speech duration (seconds)
VAD_MIN_DURATION_OFF = float(os.getenv('VAD_MIN_DURATION_OFF', '0.0'))  # Min silence to split (seconds)

# --- SECRETS ---
# Load strictly from env vars for security
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN is missing in environment variables or .env file.")