import os

# Check for Local Mode
IS_LOCAL = os.environ.get("BBAP_LOCAL_MODE") == "1"

if IS_LOCAL:
    # Local Paths
    BASE_DIR = os.getcwd()
    INPUT_DIR = os.path.join(BASE_DIR, "local_input")
    PROJECT_ROOT = os.path.join(BASE_DIR, "local_output")
    
    # Settings for Local
    # MPS for Mac, CPU otherwise, unless CUDA is available
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps" 
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    # int8 or float32 is safer for CPU/Mac than float16
    COMPUTE_TYPE = "int8" 
    
    print(f"Running in LOCAL MODE. Input: {INPUT_DIR}, Output: {PROJECT_ROOT}, Device: {DEVICE}")
else:
    # Google Colab Paths
    INPUT_DIR = "/content/drive/My Drive/Voice Record Pro"
    PROJECT_ROOT = "/content/drive/My Drive/yuzhe"
    
    # Settings for Colab T4
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"

# Output Paths (Nested in Project Root)
TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "Processed_Conversations")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "Voice_Bank", "Vectors")
LOG_FILE = os.path.join(PROJECT_ROOT, "processed_log.txt")

# Settings
CHECK_INTERVAL = 300  # seconds
SIMILARITY_THRESHOLD = 0.65  # Cosine similarity

# Supported Audio Formats
AUDIO_EXTENSIONS = ('.m4a', '.mp3', '.wav')
