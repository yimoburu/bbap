import os
import certifi
import ssl

# Fix MacOS SSL Certificate errors (Aggressive Fix)
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import config
import utils
import pipeline
import clustering
from models import ModelManager

# --- COLAB SETUP ---
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("🔹 Mounting Google Drive...")
        drive.mount('/content/drive')
except ImportError:
    pass # Assume running locally

def main():
    print(f"🚀 Starting Daily Batch Processing...")
    print(f"📂 Input: {config.INPUT_FOLDER}")
    print(f"📂 Output: {config.OUTPUT_DIR}")
    
    utils.ensure_dirs()
    
    # 1. Identify Work
    processed = utils.get_processed_files()
    
    if not os.path.exists(config.INPUT_FOLDER):
        print(f"❌ Input folder missing: {config.INPUT_FOLDER}")
        return

    # Filter files
    all_files = [f for f in os.listdir(config.INPUT_FOLDER) if f.lower().endswith(('.m4a', '.mp3', '.wav'))]
    pending_files = [f for f in all_files if f not in processed]

    if not pending_files:
        print("✅ No new files to process. System is up to date.")
        # We run maintenance anyway just in case user renamed vectors manually
        clustering.run_maintenance()
        return

    print(f"📦 Found {len(pending_files)} new files. Initializing AI Engine...")
    
    # 2. Load Models (Only if work exists)
    try:
        model_manager = ModelManager()
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    stats = {'files': 0, 'errors': 0}
    all_new_identities = set()

    # 3. Batch Loop
    for filename in pending_files:
        try:
            file_path = os.path.join(config.INPUT_FOLDER, filename)
            
            # Execute Pipeline
            new_ids = pipeline.run(file_path, filename, model_manager)
            
            # Update Stats
            all_new_identities.update(new_ids)
            stats['files'] += 1
            
            # Commit to Log
            utils.mark_as_processed(filename)
            
        except Exception as e:
            print(f"❌ Critical Error on {filename}: {e}")
            stats['errors'] += 1

    # 4. Post-Processing
    clustering.run_maintenance()
    utils.generate_daily_report(stats, all_new_identities)
    
    print("\n🎉 Batch Complete. You can close this runtime.")

if __name__ == "__main__":
    main()