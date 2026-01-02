import os
import numpy as np
import collections
from scipy.spatial.distance import cosine
from datetime import timedelta
import config
import utils

def get_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    # Cosine distance is 1 - similarity. So Similarity = 1 - distance
    return 1 - cosine(vec1, vec2)

def identify_speaker(embedding, vector_db):
    """
    Compares embedding against vector_db.
    Returns: (best_match_name, similarity_score)
    If no match > threshold, returns (None, score)
    """
    best_score = -1.0
    best_name = None
    
    for name, db_vector in vector_db.items():
        score = get_cosine_similarity(embedding, db_vector)
        if score > best_score:
            best_score = score
            best_name = name
            
    if best_score > config.SIMILARITY_THRESHOLD:
        return best_name, best_score
    else:
        return None, best_score

def process_file(file_path, model_manager):
    """
    Main processing pipeline for a single file.
    """
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}")
    
    # 1. Transcribe
    print("Transcribing...")
    result = model_manager.transcribe(file_path)
    
    # 2. Align
    print("Aligning...")
    result = model_manager.align(file_path, result)
    
    # 3. Diarize
    print("Diarizing...")
    result = model_manager.diarize(file_path, result)
    
    # 4. Identity Resolution
    print("Resolving Identities...")
    
    # Load existing vectors
    vector_db = utils.load_vectors()
    
    # Group segments by local speaker to find the longest segment for embedding extraction
    # segments are in result["segments"]
    # each segment has "speaker" key if diarization worked
    
    local_speakers = collections.defaultdict(list)
    for seg in result["segments"]:
        if "speaker" in seg:
            local_speakers[seg["speaker"]].append(seg)
            
    # Map local_speaker_id (e.g. SPEAKER_01) -> global_identity (e.g. Person_005)
    speaker_map = {}
    
    for local_spk, segments in local_speakers.items():
        # Find longest segment for this speaker
        longest_seg = max(segments, key=lambda s: s["end"] - s["start"])
        
        # Extract embedding
        print(f"Extracting embedding for {local_spk} from segment {longest_seg['start']}-{longest_seg['end']}")
        embedding = model_manager.get_embedding(file_path, longest_seg)
        # embedding is a pyannote Tensor, convert to numpy
        embedding_np = np.array(embedding)
        
        # Squeeze if needed (pyannote returns shape (1, dim) sometimes)
        if embedding_np.ndim > 1:
            embedding_np = embedding_np.flatten()
            
        # Compare with DB
        match_name, score = identify_speaker(embedding_np, vector_db)
        
        if match_name:
            print(f"Matched {local_spk} to {match_name} ({score:.2f})")
            speaker_map[local_spk] = match_name
        else:
            # Generate new ID
            all_ids = list(vector_db.keys()) + list(speaker_map.values()) # include newly assigned in this run
            new_id = utils.get_next_speaker_id(all_ids)
            print(f"No match for {local_spk}. Assigned new ID: {new_id} (Best score: {score:.2f})")
            speaker_map[local_spk] = new_id
            
            # Save new vector
            save_path = os.path.join(config.VECTOR_DB_DIR, f"{new_id}.npy")
            np.save(save_path, embedding_np)
            # Update local db representation for next speakers in this same file (unlikely to collide but good practice)
            vector_db[new_id] = embedding_np

    # 5. Output Generation
    base_timestamp = utils.parse_timestamp(filename)
    
    output_lines = []
    output_lines.append(f"Source: {filename}")
    if base_timestamp:
        output_lines.append(f"Date: {base_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("-" * 40)
    
    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        local_spk = seg.get("speaker", "Unknown")
        
        global_name = speaker_map.get(local_spk, local_spk)
        
        # Calculate display time
        if base_timestamp:
            seg_time = base_timestamp + timedelta(seconds=start)
            time_str = seg_time.strftime("%H:%M:%S")
        else:
            time_str = utils.format_timestamp(start)
            
        line = f"[{time_str}] {global_name}: {text}"
        output_lines.append(line)
        
    # Write to file
    out_filename = os.path.splitext(filename)[0] + ".txt"
    out_path = os.path.join(config.TRANSCRIPTS_DIR, out_filename)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"Saved transcript to {out_path}")
