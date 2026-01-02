import whisperx
import torch
from pyannote.audio import Model, Inference
import config

class ModelManager:
    def __init__(self):
        print("Loading models...")
        self.device = config.DEVICE
        self.compute_type = config.COMPUTE_TYPE
        
        # Load WhisperX
        # Note: HF_TOKEN must be in environment for diarization pipeline if used via whisperx,
        # but here we use pyannote explicitly for embeddings as per requirement.
        # However, whisperx also has a diarization pipeline which is very convenient.
        # The prompt asks for:
        # 1. Transcribe (WhisperX)
        # 2. Align (WhisperX)
        # 3. Diarize (Identify Speaker_00 etc) - WhisperX provides this wrapper around pyannote.
        # 4. Identity Resolution (extract embedding from pyannote).
        
        # 1. Transcription Model
        self.transcribe_model = whisperx.load_model(
            "large-v3", 
            self.device, 
            compute_type=self.compute_type
        )
        
        # 2. Diarization Pipeline (using WhisperX's wrapper for convenience)
        # This requires HF_TOKEN to be set.
        try:
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=True, # Will look for HF_TOKEN env var
                device=self.device
            )
        except Exception as e:
            print(f"Warning: Could not load DiarizationPipeline. Ensure HF_TOKEN is set. Error: {e}")
            self.diarize_model = None

        # 3. Embedding Model (Pyannote) for Identity Resolution
        # "pyannote/embedding" is deprecated in favor of "pyannote/speech-brain-..." or using Model.from_pretrained
        # We will use the standard pyannote/embedding or a compatible model for speaker verification.
        # Wespeaker-resnet34lm is a good modern choice often used with pyannote, 
        # or we use the 'pyannote/embedding' specific pipeline if available/compatible.
        # Let's use 'pyannote/wespeaker-voxceleb-resnet34-LM' or simply 'speechbrain/spkrec-ecapa-voxceleb' 
        # but the prompt specifically asked for `pyannote.audio` for speaker embedding.
        # Pyannote 3.1 uses `Model.from_pretrained("pyannote/embedding", use_auth_token=True)` but check documentation.
        # Actually standard practice now is `pyannote/wespeaker-voxceleb-resnet34-LM` or similar.
        # Given the prompt, I will stick to the standard Inference interface.
        
        try:
            self.embedding_model = Inference(
                "pyannote/embedding", 
                use_auth_token=True,
                window="whole",
                device=torch.device(self.device)
            )
            # self.embedding_model.to(torch.device(self.device)) # Inference 'device' arg handles this better in newer versions
        except Exception as e:
            print(f"Warning: Could not load Embedding Model. Ensure HF_TOKEN is set. Error: {e}")
            self.embedding_model = None

        print("Models loaded.")

    def transcribe(self, audio_file):
        """Wraps whisperx transcribe"""
        return self.transcribe_model.transcribe(audio_file, batch_size=16)

    def align(self, audio_file, result):
        """Wraps whisperx alignment"""
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=self.device
        )
        aligned_result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio_file, 
            self.device, 
            return_char_alignments=False
        )
        # clear memory
        model_a.cpu()
        del model_a
        return aligned_result

    def diarize(self, audio_file, aligned_result):
        """Wraps whisperx diarization"""
        if not self.diarize_model:
            raise RuntimeError("Diarization model not loaded.")
            
        diarize_segments = self.diarize_model(audio_file)
        
        # Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        return result

    def get_embedding(self, audio_file, segment):
        """
        Extracts embedding for a specific segment.
        segment: {'start': float, 'end': float}
        """
        # pyannote Inference (window="whole") expects a localized crop
        # We need to crop the audio in memory or pass the file with a cropping context.
        # Inference allows (file, crop=...) syntax.
        
        from pyannote.audio.core.io import Audio
        
        # We treat the input file as the source
        # crop = Segment(start, end)
        from pyannote.core import Segment
        crop = Segment(segment['start'], segment['end'])
        
        embedding = self.embedding_model(audio_file, crop=crop)
        return embedding
