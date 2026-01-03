import os
import re
import numpy as np
from datetime import datetime
import config

def ensure_dirs():
    """Create necessary project directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

def load_known_speakers():
    """Load all existing speaker vectors from the Vector DB into memory."""
    speakers = {}
    if os.path.exists(config.VECTOR_DB_DIR):
        for file in os.listdir(config.VECTOR_DB_DIR):
            if file.endswith('.npy'):
                name = os.path.splitext(file)[0]
                try:
                    speakers[name] = np.load(os.path.join(config.VECTOR_DB_DIR, file))
                except Exception as e:
                    print(f"⚠️ Corrupt vector file {file}: {e}")
    return speakers

def get_next_speaker_id(existing_names):
    """Scan existing Person_XXX IDs and find the next available number."""
    max_id = 0
    for name in existing_names:
        if name.startswith("Person_"):
            try:
                # Extract number from Person_005 -> 5
                parts = name.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    num = int(parts[1])
                    if num > max_id: max_id = num
            except:
                pass
    return f"Person_{max_id + 1:03d}"

def parse_timestamp(filename):
    """
    Extracts absolute start time from filename format: yyyymmdd-hhmmss.ext
    Example: 20231027-090000.m4a -> datetime object
    """
    match = re.search(r'(\d{8}-\d{6})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
        except ValueError:
            return None
    return None

def get_processed_files():
    """Reads the log file to see what we have already finished."""
    if not os.path.exists(config.LOG_FILE): return set()
    with open(config.LOG_FILE, 'r') as f: return set(f.read().splitlines())

def mark_as_processed(filename):
    """Appends filename to the log file."""
    with open(config.LOG_FILE, 'a') as f: f.write(filename + '\n')

def generate_daily_report(stats, new_identities):
    """Generates a Markdown briefing of the batch run."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = os.path.join(config.REPORT_DIR, f"Briefing_{date_str}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 📅 Daily Briefing: {date_str}\n\n")
        f.write("## 📊 Processing Stats\n")
        f.write(f"- Files Processed: {stats['files']}\n")
        f.write(f"- Total Errors: {stats['errors']}\n\n")
        
        f.write("## 👥 Identity Updates\n")
        if new_identities:
            f.write("The following NEW vectors were created. Please review/rename them in `Voice_Bank/Vectors`:\n")
            for identity in sorted(list(new_identities)):
                f.write(f"- 🆕 `{identity}.npy`\n")
        else:
            f.write("No new strangers detected. All speakers recognized.\n")
            
    print(f"\n📝 Daily Briefing generated: {report_path}")