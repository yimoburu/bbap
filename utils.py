import os
import glob
import re
import numpy as np
from datetime import datetime
import config

def ensure_dirs():
    """Creates necessary directories if they don't exist."""
    os.makedirs(config.TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
    # Ensure LOG_FILE exists
    if not os.path.exists(config.LOG_FILE):
        with open(config.LOG_FILE, 'w') as f:
            f.write("")

def load_processed_log():
    """Reads the processed log file into a set of filenames."""
    if not os.path.exists(config.LOG_FILE):
        return set()
    with open(config.LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def save_to_processed_log(filename):
    """Appends a filename to the processed log."""
    with open(config.LOG_FILE, 'a') as f:
        f.write(filename + "\n")

def load_vectors():
    """
    Loads all existing .npy files from the Vector DB directory into a dictionary.
    Returns:
        dict: {speaker_id (str): embedding (numpy.ndarray)}
    """
    vectors = {}
    pattern = os.path.join(config.VECTOR_DB_DIR, "*.npy")
    for file_path in glob.glob(pattern):
        # Filename example: Person_001.npy -> speaker_id: Person_001
        filename = os.path.basename(file_path)
        speaker_id = os.path.splitext(filename)[0]
        try:
            vector = np.load(file_path)
            vectors[speaker_id] = vector
        except Exception as e:
            print(f"Error loading vector {file_path}: {e}")
    return vectors

def get_next_speaker_id(existing_ids):
    """
    Generates the next available ID (e.g., Person_005) by scanning existing names.
    prefix: Person_
    """
    max_id = 0
    pattern = re.compile(r"Person_(\d+)")
    
    for sid in existing_ids:
        match = pattern.match(sid)
        if match:
            num = int(match.group(1))
            if num > max_id:
                max_id = num
    
    next_id_num = max_id + 1
    return f"Person_{next_id_num:03d}"

def parse_timestamp(filename):
    """
    Extracts date/time from filenames. 
    Assumes format contains yyyymmdd-hhmmss or similar patterns.
    Returns:
        datetime object or None if not found
    """
    # Regex for YYYYMMDD-HHMMSS (common in Voice Record Pro)
    # Example: 20231027-143000.mp4
    match = re.search(r"(\d{8})[-_](\d{6})", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return dt
        except ValueError:
            pass
            
    # Fallback: check file creation time if needed, but requirements imply filename parsing.
    # Returning None will imply relative timestamps (starting from 00:00:00) in pipeline
    return None

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS format."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
