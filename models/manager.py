"""
ModelManager: Orchestrates modular model components.
This allows easy swapping of models for better performance.
"""
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
from pyannote.audio import Audio
import config
from . import (
    WhisperXTranscriber,
    WhisperXAligner,
    PyannoteDiarizer,
    PyannoteEmbedder
)


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
    """
    Manages all AI model components.
    Uses modular components that can be easily swapped for better performance.
    """
    
    def __init__(
        self,
        transcriber=None,
        aligner=None,
        diarizer=None,
        embedder=None
    ):
        """
        Initialize ModelManager with model components.
        
        Args:
            transcriber: Transcriber instance (default: WhisperXTranscriber)
            aligner: Aligner instance (default: WhisperXAligner)
            diarizer: Diarizer instance (default: PyannoteDiarizer)
            embedder: Embedder instance (default: PyannoteEmbedder)
        """
        print("🔹 Initializing AI Engine...")
        
        # Configure WhisperX VAD options
        vad_options = {
            "vad_onset": config.WHISPERX_VAD_ONSET,
            "vad_offset": config.WHISPERX_VAD_OFFSET,
            "chunk_size": config.WHISPERX_VAD_CHUNK_SIZE,
            "min_duration_on": config.WHISPERX_VAD_MIN_DURATION_ON,
            "min_duration_off": config.WHISPERX_VAD_MIN_DURATION_OFF,
        }
        
        # Initialize model components (use defaults if not provided)
        if transcriber is None:
            print(f"   Loading WhisperX ({config.WHISPER_MODEL})...")
            self.transcriber = WhisperXTranscriber(
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                vad_options=vad_options,
                vad_method=config.WHISPERX_VAD_METHOD
            )
        else:
            self.transcriber = transcriber
        
        if aligner is None:
            self.aligner = WhisperXAligner(device=config.DEVICE)
        else:
            self.aligner = aligner
        
        if embedder is None:
            print("   Loading Pyannote Embedding...")
            self.embedder = PyannoteEmbedder(
                device=config.DEVICE,
                hf_token=config.HF_TOKEN
            )
        else:
            self.embedder = embedder
        
        if diarizer is None:
            # Diarizer is loaded on-demand (lazy loading)
            self.diarizer = PyannoteDiarizer(
                device=config.DEVICE,
                model_name=config.DIARIZATION_MODEL,
                hf_token=config.HF_TOKEN
            )
        else:
            self.diarizer = diarizer
        
        # Audio loader helper (for future use)
        self.audio_loader = Audio(sample_rate=16000, mono=True)
        self.device = config.DEVICE
        
        # Backward compatibility: expose transcription_model for pipeline
        self.transcription_model = self.transcriber
    
    def align(self, segments, audio, audio_sample_rate, language_code):
        """
        Align transcription segments to audio with word-level timestamps.
        Delegates to the aligner component.
        """
        return self.aligner.align(segments, audio, audio_sample_rate, language_code)
    
    def get_diarization_pipeline(self):
        """
        Get the diarization pipeline.
        Returns the diarizer component (for backward compatibility).
        """
        return self.diarizer
    
    def cleanup(self):
        """Force garbage collection to free GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
