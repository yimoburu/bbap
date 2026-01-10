import os
import torch
import numpy as np
import pandas as pd
import whisperx
import pyannote.core
from datetime import timedelta
import config
import utils

def run(file_path, filename, model_manager):
    print(f"   ▶️ Processing: {filename}")
    base_name = os.path.splitext(filename)[0]
    new_identities_found = []
    
    # --- 1. TRANSCRIBE (WhisperX + VAD) ---
    print("      1. Transcribing...")
    # load_audio internally resamples to 16k
    audio = whisperx.load_audio(file_path)
    
    # WhisperX performs VAD (Voice Activity Detection) here implicitly.
    # It will not transcribe long periods of silence.
    # Note: vad_options are configured in ModelManager.load_model(), not in transcribe()
    result = model_manager.transcription_model.transcribe(
        audio, 
        batch_size=config.BATCH_SIZE,
        language=config.WHISPER_LANGUAGE,
        task="transcribe",
        print_progress=True,
        verbose=True,
        combined_progress=False,
        chunk_size=config.WHISPERX_VAD_CHUNK_SIZE,

    )
    
    # --- 2. ALIGN ---
    print("      2. Aligning Timestamps...")
    audio_sample_rate = 16000  # whisperx.load_audio resamples to 16kHz
    result = model_manager.align(result["segments"], audio, audio_sample_rate, result["language"])
    
    # --- 3. DIARIZE ---
    print("      3. Diarizing Speakers...")
    diarize_model = model_manager.get_diarization_pipeline()
    
    # Prepare audio for pyannote Pipeline
    # Pyannote expects a dict with "waveform" (torch.Tensor) and "sample_rate"
    audio_tensor = torch.from_numpy(audio[None, :]).float()  # Add batch dimension: (1, samples)
    audio_dict = {
        'waveform': audio_tensor,
        'sample_rate': audio_sample_rate  # 16kHz
    }
    
    # Specify min_speakers and max_speakers to help diarization detect multiple speakers
    diarize_kwargs = {}
    if config.MIN_SPEAKERS is not None:
        diarize_kwargs['min_speakers'] = config.MIN_SPEAKERS
    if config.MAX_SPEAKERS is not None:
        diarize_kwargs['max_speakers'] = config.MAX_SPEAKERS
    
    diarization_annotation = diarize_model(audio_dict, **diarize_kwargs)
    
    # Convert pyannote Annotation to pandas DataFrame format expected by assign_word_speakers
    diarization_segments = pd.DataFrame(
        diarization_annotation.itertracks(yield_label=True),
        columns=['segment', 'label', 'speaker']
    )
    diarization_segments['start'] = diarization_segments['segment'].apply(lambda x: x.start)
    diarization_segments['end'] = diarization_segments['segment'].apply(lambda x: x.end)
    
    # Assign speaker labels (SPEAKER_01) to word segments
    final_result = whisperx.assign_word_speakers(diarization_segments, result)
    
    # Free memory immediately
    del diarize_model
    model_manager.cleanup()

    # --- 4. IDENTIFY (Vector Matching) ---
    print("      4. Resolving Identities...")
    known_speakers = utils.load_known_speakers()
    file_speaker_map = {} # Local 'SPEAKER_00' -> Global 'Person_005'
    
    # Get all unique speakers in this file
    local_speakers = set([s.get('speaker') for s in final_result["segments"] if 'speaker' in s])
    
    for local_spk in local_speakers:
        # Strategy: Find the longest continuous audio segment for this speaker
        # Longer segments = Better voice fingerprints
        spk_segments = [s for s in final_result["segments"] if s.get('speaker') == local_spk]
        best_seg = max(spk_segments, key=lambda x: x['end'] - x['start'])
        
        # Skip if segment is too short (< 0.5s) to fingerprint reliably
        if (best_seg['end'] - best_seg['start']) < 0.5: 
            continue

        try:
            # Crop audio from in-memory waveform (faster & avoids format errors)
            start_sample = int(best_seg['start'] * 16000)
            end_sample = int(best_seg['end'] * 16000)
            
            # Extract segment as tensor: (channel, time) format = (1, samples) for mono
            wav_data = audio[start_sample:end_sample]
            wav = torch.from_numpy(wav_data).float().unsqueeze(0)  # (1, samples) - correct format
            
            # Generate Embedding (Fingerprint)
            # Inference expects a dict with "waveform" (channel, time) and "sample_rate"
            audio_dict = {
                'waveform': wav,  # (1, samples) format
                'sample_rate': audio_sample_rate
            }
            # Inference returns embeddings, convert to numpy and take mean if needed
            embedding_output = model_manager.embedding_model(audio_dict)
            # Handle output - could be tensor or numpy array
            if isinstance(embedding_output, torch.Tensor):
                fingerprint = embedding_output.cpu().numpy()
            else:
                fingerprint = np.array(embedding_output)
            # If output has batch dimension, take mean
            if len(fingerprint.shape) > 1:
                fingerprint = fingerprint.mean(axis=0)
            
            # Compare against known database
            best_match = None
            best_score = -1.0
            
            for name, ref_vector in known_speakers.items():
                # Cosine Similarity Formula
                score = np.dot(fingerprint, ref_vector) / (np.linalg.norm(fingerprint) * np.linalg.norm(ref_vector))
                if score > best_score:
                    best_score = score
                    best_match = name
            
            # Decision Logic
            if best_score > config.SIMILARITY_THRESHOLD:
                # MATCH FOUND
                file_speaker_map[local_spk] = best_match
                print(f"         🔗 Matched {local_spk} -> {best_match} ({best_score:.2f})")
            else:
                # NEW SPEAKER
                new_id = utils.get_next_speaker_id(known_speakers.keys())
                
                # Save to disk
                np.save(os.path.join(config.VECTOR_DB_DIR, f"{new_id}.npy"), fingerprint)
                
                # Update in-memory cache so future segments in this file match this new ID
                known_speakers[new_id] = fingerprint
                file_speaker_map[local_spk] = new_id
                
                new_identities_found.append(new_id)
                print(f"         🆕 New Identity: {local_spk} -> {new_id}")
                
        except Exception as e:
            print(f"         ⚠️ ID Error for {local_spk}: {e}")

    # --- 5. FORMAT & SAVE ---
    start_dt = utils.parse_timestamp(filename)
    formatted_transcript = []
    
    for segment in final_result["segments"]:
        # Calculate Time
        if start_dt:
            # Absolute Wall-Clock Time with date
            s_time = start_dt + timedelta(seconds=segment['start'])
            start = s_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Relative Time (fallback if no date in filename)
            start = timedelta(seconds=int(segment['start']))
            
        raw_speaker = segment.get('speaker', 'Unknown')
        final_name = file_speaker_map.get(raw_speaker, raw_speaker)
        text = segment['text'].strip()
        
        formatted_transcript.append(f"[{start}] {final_name}: {text}")

    # Write text file
    with open(os.path.join(config.OUTPUT_DIR, f"{base_name}.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(formatted_transcript))
    
    print(f"      ✅ Saved Transcript: {base_name}.txt")
    return new_identities_found