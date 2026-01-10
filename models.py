import whisperx
import whisperx.diarize
from pyannote.audio import Inference, Audio, Model, Pipeline
import pandas as pd
import config
import torch
import gc
import omegaconf
import typing
import collections
import argparse
import pyannote.audio.core.model
import pyannote.audio.core.task
import pytorch_lightning.callbacks.early_stopping
import pytorch_lightning.callbacks.model_checkpoint
import numpy as np
import os
import tempfile


# Pytorch 2.6+ security fix: Allow loading Omegaconf objects
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig, 
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
    list,
    dict,
    set,
    int,
    float,
    str,
    collections.defaultdict,
    collections.OrderedDict,
    argparse.Namespace,
    omegaconf.nodes.AnyNode,
    omegaconf.base.Metadata,
    omegaconf.nodes.ValueNode,
    omegaconf.nodes.StringNode,
    omegaconf.nodes.IntegerNode,
    omegaconf.nodes.FloatNode,
    omegaconf.nodes.BooleanNode,
    omegaconf.nodes.EnumNode,
    torch.torch_version.TorchVersion,
    pyannote.audio.core.model.Introspection,
    pyannote.audio.core.task.Specifications,
    pyannote.audio.core.task.Problem,
    pyannote.audio.core.task.Resolution,
    pytorch_lightning.callbacks.early_stopping.EarlyStopping,
    pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
])

class ModelManager:
    def __init__(self):
        print("🔹 Initializing AI Engine...")
        
        # 1. Load Whisper (Transcription)
        # Large-v3 handles code-switching (mixed EN/CN) best.
        print(f"   Loading WhisperX ({config.WHISPER_MODEL})...")
        
        # Configure WhisperX VAD options for better segmentation
        # Lower thresholds = more sensitive (detects more segments)
        # Smaller chunk_size = more segments (less merging of VAD segments)
        vad_options = {
            "vad_onset": config.WHISPERX_VAD_ONSET,  # Default: 0.500, lower = more sensitive
            "vad_offset": config.WHISPERX_VAD_OFFSET,  # Default: 0.363, lower = more sensitive
            "chunk_size": config.WHISPERX_VAD_CHUNK_SIZE,  # Default: 30s, smaller = more segments
            # Note: min_duration_on and min_duration_off are passed but may not be used
            # by WhisperX's internal merge_chunks function (currently hardcoded to 0.1)
            "min_duration_on": config.WHISPERX_VAD_MIN_DURATION_ON,  # Min speech duration (seconds)
            "min_duration_off": config.WHISPERX_VAD_MIN_DURATION_OFF,  # Min silence to split (seconds)
        }
        
        # VAD method: "pyannote" (default, more accurate) or "silero" (faster)
        vad_method = config.WHISPERX_VAD_METHOD
        
        self.transcription_model = whisperx.load_model(
            config.WHISPER_MODEL, 
            config.DEVICE, 
            compute_type=config.COMPUTE_TYPE,
            vad_options=vad_options,
            vad_method=vad_method
        )
        
        # 2. Load Embedding (Fingerprinting)
        # Pyannote embedding extracts the mathematical vector from audio
        print("   Loading Pyannote Embedding...")

        model = Model.from_pretrained("pyannote/embedding", use_auth_token=config.HF_TOKEN)
        # Convert device string to torch.device if needed (Inference expects torch.device, not string)
        if isinstance(config.DEVICE, str):
            device = torch.device(config.DEVICE)
        else:
            device = config.DEVICE
        self.embedding_model = Inference(model, device=device)

        # # inference on the whole file
        # inference("file.wav")

        # # inference on an excerpt
        # from pyannote.core import Segment
        # excerpt = Segment(start=2.0, end=5.0)
        # inference.crop("file.wav", excerpt)
        
        # Audio loader helper
        self.audio_loader = Audio(sample_rate=16000, mono=True)
        self.device = config.DEVICE

    def load_align_model(self, language_code):
        """Loads alignment model specific to the detected language."""
        return whisperx.load_align_model(
            language_code=language_code, 
            device=self.device
        )
    
    def align(self, segments, audio, audio_sample_rate, language_code):
        """
        Align transcription segments to audio with word-level timestamps.
        Uses WhisperX with character-level alignment for Chinese (better accuracy).
        Fixes word_segments timestamps during alignment to prevent misalignment.
        """
        detected_language = language_code.lower() if language_code else ""
        is_chinese = detected_language in ["zh", "zh-cn", "zh-tw", "chinese"]
        
        # For Chinese, use character-level alignment for better accuracy
        # Chinese doesn't have spaces between words, so character-level alignment is more precise
        use_char_alignments = is_chinese
        
        if is_chinese:
            print("      Using character-level WhisperX alignment for Chinese")
        
        # Alignment with integrated timestamp fixing
        result = self._align_with_whisperx(segments, audio, language_code, use_char_alignments, original_segments=segments)
        
        return result
    
    def _align_with_whisperx(self, segments, audio, language_code, use_char_alignments=False, original_segments=None):
        """
        Internal method to align using WhisperX.
        Fixes timestamp misalignment within segments by redistributing timestamps
        based on character position in the text.
        
        Note: whisperx.align function signature:
        whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
        
        Issue: Words are in the correct segment, but timestamps are wrong.
        Solution: Redistribute timestamps within each segment based on character position.
        """
        model_a, metadata = self.load_align_model(language_code)
        
        # Pre-process segments to improve alignment accuracy
        # Filter empty segments and validate text
        processed_segments = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if text and seg.get('start', 0) < seg.get('end', 0):
                # Normalize text: remove extra spaces, normalize punctuation
                text = ' '.join(text.split())  # Normalize whitespace
                seg_copy = seg.copy()
                seg_copy['text'] = text
                processed_segments.append(seg_copy)
        
        if not processed_segments:
            # Return empty result if no valid segments
            return {'segments': [], 'word_segments': []}
        
        # For Chinese, ensure we're using character-level alignment
        # This is critical for accurate timestamps with Chinese characters
        # Character-level alignment is more precise for languages without word boundaries
        result = whisperx.align(
            processed_segments, 
            model_a, 
            metadata, 
            audio, 
            self.device, 
            return_char_alignments=use_char_alignments
        )
        
        # Fix timestamps within segments - redistribute based on character position
        # This addresses the issue where words are in the correct segment but timestamps are wrong
        result = self._fix_timestamps_within_segments(result)
        
        # Free memory immediately
        del model_a
        self.cleanup()
        return result
    
    def _fix_timestamps_within_segments(self, result):
        """
        Fix timestamps within segments by redistributing them based on character position.
        This addresses cases where words are in the correct segment but have wrong timestamps.
        """
        if 'segments' not in result:
            return result
        
        aligned_segments = result.get('segments', [])
        if not aligned_segments:
            return result
        
        # Process each segment independently
        for seg in aligned_segments:
            words = seg.get('words', [])
            if not words:
                continue
            
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            seg_text = seg.get('text', '').strip()
            
            if not seg_text or seg_end <= seg_start:
                continue
            
            # Calculate character positions for each word in the segment text
            seg_duration = seg_end - seg_start
            total_chars = len(seg_text)
            
            if total_chars == 0:
                continue
            
            # Find character positions for each word
            char_positions = []
            current_pos = 0
            
            for word in words:
                word_text = word.get('word', '').strip()
                if not word_text:
                    continue
                
                # Find word position in segment text
                # Remove spaces from word_text for matching (Chinese doesn't have spaces)
                word_text_no_space = word_text.replace(' ', '')
                seg_text_no_space = seg_text.replace(' ', '')
                
                # Try to find the word in the text starting from current position
                word_start_in_text = seg_text_no_space.find(word_text_no_space, current_pos)
                
                if word_start_in_text == -1:
                    # Word not found, use current position
                    word_start_in_text = current_pos
                
                word_end_in_text = word_start_in_text + len(word_text_no_space)
                current_pos = word_end_in_text
                
                char_positions.append({
                    'word': word,
                    'char_start': word_start_in_text,
                    'char_end': word_end_in_text,
                    'word_text': word_text
                })
            
            # Redistribute timestamps based on character position
            for char_info in char_positions:
                word = char_info['word']
                char_start = char_info['char_start']
                char_end = char_info['char_end']
                
                # Calculate timestamp based on character position
                # Distribute time proportionally within the segment
                word_start = seg_start + (char_start / total_chars) * seg_duration
                word_end = seg_start + (char_end / total_chars) * seg_duration
                
                # Ensure minimum duration
                if word_end <= word_start:
                    word_end = word_start + 0.05
                
                # Update word timestamps in-place
                word['start'] = word_start
                word['end'] = word_end
        
        # Reconstruct word_segments from fixed segments
        word_segments = []
        for seg in aligned_segments:
            words = seg.get('words', [])
            for word in words:
                if word.get('word', '').strip():
                    word_segments.append({
                        'word': word.get('word', ''),
                        'start': word.get('start', seg.get('start', 0)),
                        'end': word.get('end', seg.get('end', 0)),
                        'score': word.get('score', 1.0)
                    })
        
        # Sort by start time
        word_segments.sort(key=lambda w: w.get('start', 0))
        result['word_segments'] = word_segments
        
        return result
        
    def get_diarization_pipeline(self):
        """Loads the pyannote speaker diarization pipeline directly."""
        print(f"   Loading Pyannote Diarization Pipeline ({config.DIARIZATION_MODEL})...")
        # Convert device string to torch.device if needed
        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device
        
        pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.HF_TOKEN
        ).to(device)
        return pipeline

    def cleanup(self):
        """Force garbage collection to free GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
