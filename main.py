import os
import time
import utils
import config
from models import ModelManager
from pipeline import process_file

def main():
    print("Starting Black Box Audio Protocol...")
    
    # 1. Setup & Checks
    if not config.IS_LOCAL:
        # 1a. Mount Drive (Only in Colab)
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            print("Not running in Colab, skipping Drive mount.")

    if not os.path.exists(config.INPUT_DIR):
        print(f"Warning: Input directory {config.INPUT_DIR} does not exist.")
        # In a real deployed scenario we might want to wait or error out.
        # For now, we'll try to create it? No, it's google drive input.
        # We should just wait until it exists or warn user.
        print("Please ensure Google Drive is mounted and path is correct.")
        
    utils.ensure_dirs()
    
    # 2. Init Models
    try:
        model_mgr = ModelManager()
    except Exception as e:
        print(f"Critical Error loading models: {e}")
        return

    print(f"Monitoring {config.INPUT_DIR} every {config.CHECK_INTERVAL} seconds...")

    # 3. Watchdog Loop
    while True:
        try:
            processed_files = utils.load_processed_log()
            
            # Scan directory
            if os.path.exists(config.INPUT_DIR):
                files = [
                    os.path.join(config.INPUT_DIR, f) 
                    for f in os.listdir(config.INPUT_DIR) 
                    if f.lower().endswith(config.AUDIO_EXTENSIONS)
                ]
                
                # Sort by modification time to process oldest first? Or just name.
                # Let's sort by name for stability.
                files.sort()
                
                for file_path in files:
                    filename = os.path.basename(file_path)
                    
                    if filename not in processed_files:
                        print(f"Found new file: {filename}")
                        try:
                            # Run Pipeline
                            process_file(file_path, model_mgr)
                            
                            # Mark done
                            utils.save_to_processed_log(filename)
                            print(f"Completed {filename}")
                            
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
                            # We might want to log this to an error log file
            else:
                print(f"Input dir not accessible yet...")

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Unexpected error in watchdog loop: {e}")
            
        print(f"Sleeping for {config.CHECK_INTERVAL}s...")
        time.sleep(config.CHECK_INTERVAL)

if __name__ == "__main__":
    main()
