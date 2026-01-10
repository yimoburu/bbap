"""
WhisperX model implementations for transcription and alignment.
"""
import whisperx
import numpy as np
from typing import Dict, List, Any, Optional
import config
from .base import Transcriber, Aligner


class WhisperXTranscriber(Transcriber):
    """WhisperX-based transcription implementation."""
    
    def __init__(self, device: str, compute_type: str, vad_options: Dict, vad_method: str):
        """
        Initialize WhisperX transcription model.
        
        Args:
            device: Device string ('cpu', 'cuda', 'mps')
            compute_type: Compute type ('float16', 'int8', 'float32')
            vad_options: VAD configuration dictionary
            vad_method: VAD method ('pyannote' or 'silero')
        """
        self.device = device
        self.compute_type = compute_type
        self.vad_options = vad_options
        self.vad_method = vad_method
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the WhisperX transcription model."""
        self.model = whisperx.load_model(
            config.WHISPER_MODEL,
            self.device,
            compute_type=self.compute_type,
            vad_options=self.vad_options,
            vad_method=self.vad_method
        )
    
    def transcribe(
        self,
        audio: np.ndarray,
        batch_size: int = 16,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe audio using WhisperX."""
        return self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
            task=task,
            combined_progress=False,
            chunk_size=config.WHISPERX_VAD_CHUNK_SIZE,
            **kwargs
        )


class WhisperXAligner(Aligner):
    """WhisperX-based alignment implementation with Chinese character support."""
    
    def __init__(self, device: str):
        """
        Initialize WhisperX aligner.
        
        Args:
            device: Device string ('cpu', 'cuda', 'mps')
        """
        self.device = device
    
    def load_align_model(self, language_code: str):
        """Load alignment model for specific language."""
        return whisperx.load_align_model(
            language_code=language_code,
            device=self.device
        )
    
    def align(
        self,
        segments: List[Dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        language_code: str
    ) -> Dict[str, Any]:
        """
        Align transcription segments with word/character-level timestamps.
        Uses character-level alignment for Chinese languages.
        """
        detected_language = language_code.lower() if language_code else ""
        is_chinese = detected_language in ["zh", "zh-cn", "zh-tw", "chinese"]
        use_char_alignments = is_chinese
        
        if is_chinese:
            print("      Using character-level WhisperX alignment for Chinese")
        
        return self._align_with_whisperx(
            segments, audio, language_code, use_char_alignments
        )
    
    def _align_with_whisperx(
        self,
        segments: List[Dict[str, Any]],
        audio: np.ndarray,
        language_code: str,
        use_char_alignments: bool = False
    ) -> Dict[str, Any]:
        """Internal alignment method using WhisperX."""
        model_a, metadata = self.load_align_model(language_code)
        
        # Pre-process segments: filter empty and normalize text
        processed_segments = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if text and seg.get('start', 0) < seg.get('end', 0):
                text = ' '.join(text.split())  # Normalize whitespace
                seg_copy = seg.copy()
                seg_copy['text'] = text
                processed_segments.append(seg_copy)
        
        if not processed_segments:
            return {'segments': [], 'word_segments': []}
        
        # Perform alignment
        result = whisperx.align(
            processed_segments,
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=use_char_alignments
        )
        
        # Fix timestamps within segments (for Chinese character alignment)
        result = self._fix_timestamps_within_segments(result)
        
        # Cleanup
        del model_a
        return result
    
    def _fix_timestamps_within_segments(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix timestamps within segments by redistributing based on character position.
        This addresses timestamp drift issues, especially for Chinese.
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
                
                # Find word position in segment text (handle spaces)
                word_text_no_space = word_text.replace(' ', '')
                seg_text_no_space = seg_text.replace(' ', '')
                
                word_start_in_text = seg_text_no_space.find(word_text_no_space, current_pos)
                if word_start_in_text == -1:
                    word_start_in_text = current_pos
                
                word_end_in_text = word_start_in_text + len(word_text_no_space)
                current_pos = word_end_in_text
                
                char_positions.append({
                    'word': word,
                    'char_start': word_start_in_text,
                    'char_end': word_end_in_text,
                })
            
            # Redistribute timestamps based on character position
            for char_info in char_positions:
                word = char_info['word']
                char_start = char_info['char_start']
                char_end = char_info['char_end']
                
                # Calculate timestamp proportionally
                word_start = seg_start + (char_start / total_chars) * seg_duration
                word_end = seg_start + (char_end / total_chars) * seg_duration
                
                # Ensure minimum duration
                if word_end <= word_start:
                    word_end = word_start + 0.05
                
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
