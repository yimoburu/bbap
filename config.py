import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- PATH CONFIGURATION ---
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
DEVICE = os.getenv('DEVICE', 'cuda')
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', 'float16')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))

# --- SECRETS ---
# Load strictly from env vars for security
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN is missing in environment variables or .env file.")