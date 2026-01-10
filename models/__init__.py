"""
Model package for BBAP system.
Provides modular model components that can be easily swapped.
"""
from .base import Transcriber, Aligner, Diarizer, Embedder
from .whisperx_models import WhisperXTranscriber, WhisperXAligner
from .pyannote_models import PyannoteDiarizer, PyannoteEmbedder
from .manager import ModelManager

__all__ = [
    'Transcriber',
    'Aligner',
    'Diarizer',
    'Embedder',
    'WhisperXTranscriber',
    'WhisperXAligner',
    'PyannoteDiarizer',
    'PyannoteEmbedder',
    'ModelManager',
]
