"""
Abstract base classes for model components.
This allows easy swapping of models for better performance.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
import torch


class Transcriber(ABC):
    """Abstract base class for transcription models."""
    
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        batch_size: int = 16,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text with segment-level timestamps.
        
        Args:
            audio: Audio array (mono, 16kHz)
            batch_size: Batch size for processing
            language: Language code (None for auto-detect)
            task: "transcribe" or "translate"
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with:
                - 'segments': List of segment dicts with 'start', 'end', 'text'
                - 'language': Detected language code
        """
        pass


class Aligner(ABC):
    """Abstract base class for alignment models."""
    
    @abstractmethod
    def align(
        self,
        segments: List[Dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        language_code: str
    ) -> Dict[str, Any]:
        """
        Align transcription segments to word/character-level timestamps.
        
        Args:
            segments: List of segment dicts from transcription
            audio: Audio array (mono, 16kHz)
            sample_rate: Sample rate (typically 16000)
            language_code: Language code for alignment model
            
        Returns:
            Dictionary with aligned segments (same structure, with 'words' added):
                - 'segments': List of segment dicts with word-level timestamps
                - 'word_segments': Flat list of word dicts (optional)
        """
        pass


class Diarizer(ABC):
    """Abstract base class for speaker diarization models."""
    
    @abstractmethod
    def diarize(
        self,
        audio_dict: Dict[str, torch.Tensor],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio_dict: Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Diarization result (format depends on implementation, but should be convertible
            to pandas DataFrame with columns: 'segment', 'label', 'speaker', 'start', 'end')
        """
        pass


class Embedder(ABC):
    """Abstract base class for voice embedding models."""
    
    @abstractmethod
    def embed(
        self,
        audio_dict: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        Generate voice embedding vector from audio segment.
        
        Args:
            audio_dict: Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
            
        Returns:
            1D NumPy array representing the voice embedding
        """
        pass
