import whisperx
from pyannote.audio import Inference, Audio
import config
import torch
import gc

class ModelManager:
    def __init__(self):
        print("🔹 Initializing AI Engine...")
        
        # 1. Load Whisper (Transcription)
        # Large-v3 handles code-switching (mixed EN/CN) best.
        print("   Loading WhisperX Large-v3...")
        self.transcription_model = whisperx.load_model(
            "large-v3", 
            config.DEVICE, 
            compute_type=config.COMPUTE_TYPE
        )
        
        # 2. Load Embedding (Fingerprinting)
        # Pyannote embedding extracts the mathematical vector from audio
        print("   Loading Pyannote Embedding...")
        self.embedding_model = Inference(
            "pyannote/embedding", 
            use_auth_token=config.HF_TOKEN, 
            device=config.DEVICE
        )
        
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
        return whisperx.DiarizationPipeline(
            use_auth_token=config.HF_TOKEN, 
            device=self.device
        )

    def cleanup(self):
        """Force garbage collection to free GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
