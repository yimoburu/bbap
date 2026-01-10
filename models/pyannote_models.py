"""
Pyannote model implementations for diarization and embedding.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pyannote.audio import Inference, Audio, Model, Pipeline
import config
from .base import Diarizer, Embedder


class PyannoteDiarizer(Diarizer):
    """Pyannote-based speaker diarization implementation."""
    
    def __init__(self, device: str, model_name: str, hf_token: str):
        """
        Initialize Pyannote diarization pipeline.
        
        Args:
            device: Device string ('cpu', 'cuda', 'mps')
            model_name: Model identifier (e.g., 'pyannote/speaker-diarization-3.1')
            hf_token: Hugging Face authentication token
        """
        self.device = self._convert_device(device)
        self.model_name = model_name
        self.hf_token = hf_token
        self._pipeline = None
    
    def _convert_device(self, device: str) -> torch.device:
        """Convert device string to torch.device object."""
        if isinstance(device, str):
            return torch.device(device)
        return device
    
    def get_pipeline(self) -> Pipeline:
        """Get or create the diarization pipeline (lazy loading)."""
        if self._pipeline is None:
            print(f"   Loading Pyannote Diarization Pipeline ({self.model_name})...")
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            ).to(self.device)
        return self._pipeline
    
    def diarize(
        self,
        audio_dict: Dict[str, torch.Tensor],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform speaker diarization on audio.
        
        Returns:
            pandas DataFrame with columns: 'segment', 'label', 'speaker', 'start', 'end'
        """
        pipeline = self.get_pipeline()
        
        # Prepare diarization kwargs
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarize_kwargs['max_speakers'] = max_speakers
        diarize_kwargs.update(kwargs)
        
        # Run diarization
        diarization_annotation = pipeline(audio_dict, **diarize_kwargs)
        
        # Convert pyannote Annotation to pandas DataFrame
        diarization_segments = pd.DataFrame(
            diarization_annotation.itertracks(yield_label=True),
            columns=['segment', 'label', 'speaker']
        )
        diarization_segments['start'] = diarization_segments['segment'].apply(lambda x: x.start)
        diarization_segments['end'] = diarization_segments['segment'].apply(lambda x: x.end)
        
        return diarization_segments
    
    def cleanup(self):
        """Free memory by deleting the pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None


class PyannoteEmbedder(Embedder):
    """Pyannote-based voice embedding implementation."""
    
    def __init__(self, device: str, hf_token: str):
        """
        Initialize Pyannote embedding model.
        
        Args:
            device: Device string ('cpu', 'cuda', 'mps')
            hf_token: Hugging Face authentication token
        """
        self.device = self._convert_device(device)
        self.hf_token = hf_token
        self._model = None
        self._inference = None
        self._load_model()
    
    def _convert_device(self, device: str) -> torch.device:
        """Convert device string to torch.device object."""
        if isinstance(device, str):
            return torch.device(device)
        return device
    
    def _load_model(self):
        """Load the Pyannote embedding model."""
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=self.hf_token)
        self._inference = Inference(model, device=self.device)
    
    def embed(self, audio_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Generate voice embedding vector from audio segment.
        
        Args:
            audio_dict: Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
            
        Returns:
            1D NumPy array representing the voice embedding
        """
        # Run inference
        embedding_output = self._inference(audio_dict)
        
        # Convert to numpy array
        if isinstance(embedding_output, torch.Tensor):
            fingerprint = embedding_output.cpu().numpy()
        else:
            fingerprint = np.array(embedding_output)
        
        # If output has batch dimension, take mean
        if len(fingerprint.shape) > 1:
            fingerprint = fingerprint.mean(axis=0)
        
        # Ensure 1D array
        return fingerprint.flatten()
