import whisperx
import whisperx.diarize
from pyannote.audio import Inference, Audio, Model
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
        self.transcription_model = whisperx.load_model(
            config.WHISPER_MODEL, 
            config.DEVICE, 
            compute_type=config.COMPUTE_TYPE
        )
        
        # 2. Load Embedding (Fingerprinting)
        # Pyannote embedding extracts the mathematical vector from audio
        print("   Loading Pyannote Embedding...")

        model = Model.from_pretrained("pyannote/embedding", use_auth_token=config.HF_TOKEN)
        self.embedding_model = Inference(model, device=config.DEVICE)

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
        
    def get_diarization_pipeline(self):
        """Loads the speaker separation pipeline."""
        return whisperx.diarize.DiarizationPipeline(
            use_auth_token=config.HF_TOKEN, 
            device=self.device
        )

    def cleanup(self):
        """Force garbage collection to free GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
