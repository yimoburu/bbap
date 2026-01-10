# BBAP Models Architecture Specification

**Version:** 1.0  
**Last Updated:** 2025-01-09  
**Purpose:** Design specification for the modular model architecture in BBAP. This document defines the architecture, interfaces, and requirements for all model components. It serves as the single source of truth for model design and can be used to reimplement the models layer.

---

## 1. Architecture Overview

### 1.1 Design Philosophy
The models architecture follows a **modular, pluggable design** that allows easy swapping of model implementations without changing the rest of the system. This enables:
- Performance improvements by switching to better models
- Model experimentation without system-wide changes
- Support for multiple model backends
- Clear separation of concerns

### 1.2 Component Architecture
```
┌─────────────────────────────────────────┐
│         ModelManager (Orchestrator)      │
│  - Coordinates all model components     │
│  - Provides unified interface           │
│  - Handles initialization & cleanup    │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│ Transcriber │  │   Aligner   │
│  (Abstract) │  │  (Abstract) │
└─────────────┘  └─────────────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Diarizer  │  │  Embedder   │
│  (Abstract) │  │  (Abstract) │
└─────────────┘  └─────────────┘
```

### 1.3 Model Component Types
The system requires four distinct model component types:
1. **Transcriber**: Converts audio to text with segment-level timestamps
2. **Aligner**: Refines timestamps to word/character level
3. **Diarizer**: Identifies "who spoke when" in multi-speaker audio
4. **Embedder**: Generates voice embeddings for speaker identification

---

## 2. Abstract Base Interfaces

### 2.1 Transcriber Interface

**Purpose**: Convert audio to text with segment-level timestamps.

**Required Method**:
```python
def transcribe(
    audio: np.ndarray,
    batch_size: int = 16,
    language: Optional[str] = None,
    task: str = "transcribe",
    **kwargs
) -> Dict[str, Any]
```

**Input Specifications**:
- `audio`: NumPy array, mono, 16kHz sample rate, shape `(samples,)`
- `batch_size`: Integer, configurable (default: 16)
- `language`: Optional string, language code (e.g., `"zh"`, `"en"`) or `None` for auto-detect
- `task`: String, `"transcribe"` or `"translate"`

**Output Format**:
```python
{
    "segments": [
        {
            "start": float,  # seconds, segment start time
            "end": float,    # seconds, segment end time
            "text": str      # transcribed text
        },
        ...
    ],
    "language": str  # detected language code (e.g., "zh", "en")
}
```

**Requirements**:
- Must perform Voice Activity Detection (VAD) to skip silence
- Must support configurable VAD parameters
- Must handle multi-speaker audio (may produce single segment initially)
- Must support language auto-detection
- Audio must be resampled to 16kHz mono before processing

### 2.2 Aligner Interface

**Purpose**: Refine transcription timestamps to word/character-level precision.

**Required Method**:
```python
def align(
    segments: List[Dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    language_code: str
) -> Dict[str, Any]
```

**Input Specifications**:
- `segments`: List of segment dictionaries from transcription stage
- `audio`: NumPy array, mono, 16kHz, shape `(samples,)`
- `sample_rate`: Integer, must be 16000
- `language_code`: String, language code from transcription

**Output Format**:
```python
{
    "segments": [
        {
            "start": float,
            "end": float,
            "text": str,
            "words": [
                {
                    "word": str,
                    "start": float,  # seconds
                    "end": float     # seconds
                },
                ...
            ]
        },
        ...
    ],
    "word_segments": [  # Optional: flat list of all words
        {
            "word": str,
            "start": float,
            "end": float,
            "score": float
        },
        ...
    ],
    "language": str
}
```

**Critical Requirements**:
- **Chinese Language Support**: Must use character-level alignment for Chinese (`zh`, `zh-cn`, `zh-tw`)
- **Timestamp Accuracy**: Must prevent timestamp drift, especially for character-based languages
- **Empty Segment Handling**: Must filter and validate segments before alignment
- **Timestamp Redistribution**: For Chinese, must redistribute timestamps based on character position to prevent drift

**Language-Specific Behavior**:
- Chinese languages: Use character-level alignment (`return_char_alignments=True`)
- Other languages: Use word-level alignment
- Timestamp fixing algorithm must account for character position in text

### 2.3 Diarizer Interface

**Purpose**: Identify "who spoke when" in multi-speaker audio.

**Required Method**:
```python
def diarize(
    audio_dict: Dict[str, torch.Tensor],
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

**Input Specifications**:
- `audio_dict`: Dictionary with required keys:
  - `'waveform'`: torch.Tensor, shape `(1, samples)` for mono audio
  - `'sample_rate'`: int, must be 16000
- `min_speakers`: Optional integer, minimum number of speakers
- `max_speakers`: Optional integer, maximum number of speakers

**Output Format**:
```python
pandas.DataFrame with columns:
- 'segment': pyannote.core.Segment (or compatible object)
- 'label': str (usually empty)
- 'speaker': str (e.g., 'SPEAKER_00', 'SPEAKER_01')
- 'start': float (seconds)
- 'end': float (seconds)
```

**Requirements**:
- Must accept audio in Pyannote-compatible format
- Must support optional speaker count constraints
- Output must be convertible to pandas DataFrame
- Speaker labels must be in format `SPEAKER_XX` where XX is zero-padded
- Must handle lazy loading (pipeline created on-demand)

**Memory Management**:
- Diarization pipeline should be loaded on-demand
- Must provide `cleanup()` method to free memory
- Pipeline should be deleted after use

### 2.4 Embedder Interface

**Purpose**: Generate voice embedding vectors from audio segments for speaker identification.

**Required Method**:
```python
def embed(
    audio_dict: Dict[str, torch.Tensor]
) -> np.ndarray
```

**Input Specifications**:
- `audio_dict`: Dictionary with required keys:
  - `'waveform'`: torch.Tensor, shape `(1, samples)` for mono audio
  - `'sample_rate'`: int, must be 16000

**Output Format**:
- 1D NumPy array (flattened embedding vector)
- Shape: `(embedding_dim,)` where embedding_dim is model-specific

**Requirements**:
- Must accept audio in Pyannote-compatible format
- Output must be a 1D NumPy array
- If model outputs multi-dimensional array, must flatten or take mean
- Embedding should be normalized (implementation-dependent)

**Use Case**:
- Used for speaker identity matching via cosine similarity
- Input audio segments should be ≥ 0.5 seconds for reliable embeddings

---

## 3. ModelManager Specification

### 3.1 Purpose
The `ModelManager` orchestrates all model components and provides a unified interface to the pipeline. It handles initialization, component coordination, and resource cleanup.

### 3.2 Interface

#### `ModelManager.__init__(transcriber=None, aligner=None, diarizer=None, embedder=None)`

**Purpose**: Initialize ModelManager with model components.

**Parameters**:
- `transcriber`: Optional Transcriber instance (default: creates WhisperXTranscriber)
- `aligner`: Optional Aligner instance (default: creates WhisperXAligner)
- `diarizer`: Optional Diarizer instance (default: creates PyannoteDiarizer)
- `embedder`: Optional Embedder instance (default: creates PyannoteEmbedder)

**Behavior**:
- If component is `None`, creates default implementation
- Loads models into memory during initialization
- Prints progress messages
- Handles device configuration (CPU/CUDA/MPS)

**Side Effects**:
- Loads models into memory
- Configures device settings
- Sets up PyTorch security allowlist (for PyTorch 2.6+)

#### `ModelManager.align(segments, audio, audio_sample_rate, language_code)`

**Purpose**: Align transcription segments to word/character level.

**Parameters**:
- `segments`: List of segment dictionaries from transcription
- `audio`: NumPy array (mono, 16kHz)
- `audio_sample_rate`: int (must be 16000)
- `language_code`: str (language code from transcription)

**Returns**: Dictionary with aligned segments (see Aligner output format)

**Delegation**: Delegates to the `aligner` component.

#### `ModelManager.get_diarization_pipeline()`

**Purpose**: Get the diarization pipeline (for backward compatibility).

**Returns**: Diarizer instance

**Note**: Returns the diarizer component directly. Pipeline creation is handled internally.

#### `ModelManager.cleanup()`

**Purpose**: Free memory and run garbage collection.

**Side Effects**:
- Runs `gc.collect()`
- Clears CUDA cache if available
- Should be called after processing to free GPU memory

### 3.3 Backward Compatibility
- Exposes `transcription_model` attribute pointing to transcriber (for legacy code)
- Maintains same method signatures as previous implementation

---

## 4. Data Format Specifications

### 4.1 Audio Input Format

**For Transcription & Alignment**:
- Format: NumPy array
- Shape: `(samples,)` for mono audio
- Sample Rate: 16000 Hz
- Type: `np.ndarray` (float32 or float64)

**For Diarization & Embedding**:
- Format: Dictionary
- Required keys:
  ```python
  {
      'waveform': torch.Tensor,  # Shape: (1, samples) for mono
      'sample_rate': int          # Must be 16000
  }
  ```
- Tensor dtype: `torch.float32`
- Tensor device: Must match model device (CPU/CUDA/MPS)

### 4.2 Device Configuration

**Device Types**:
- `"cpu"`: CPU processing
- `"cuda"`: NVIDIA GPU (CUDA)
- `"mps"`: Apple Silicon GPU (Metal Performance Shaders)

**Device Conversion**:
- Pyannote models require `torch.device` objects, not strings
- ModelManager must convert string device names to `torch.device`
- Auto-detection priority: CUDA > MPS > CPU

**Compute Types**:
- `"float16"`: GPU (CUDA) - default for GPU
- `"int8"`: CPU/MPS - default for CPU/MPS
- `"float32"`: Full precision (slower)

---

## 5. Configuration Requirements

### 5.1 Model Configuration Parameters

**Transcription Model**:
- `WHISPER_MODEL`: Model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `WHISPER_LANGUAGE`: Language code or `None` for auto-detect
- `BATCH_SIZE`: Batch size for processing (default: 16)
- `WHISPERX_VAD_METHOD`: `"pyannote"` (accurate) or `"silero"` (faster)
- `WHISPERX_VAD_ONSET`: Speech start threshold 0.0-1.0 (default: 0.5)
- `WHISPERX_VAD_OFFSET`: Speech end threshold 0.0-1.0 (default: 0.363)
- `WHISPERX_VAD_CHUNK_SIZE`: Max chunk size in seconds (default: 5)
- `WHISPERX_VAD_MIN_DURATION_ON`: Min speech duration (default: 0.0)
- `WHISPERX_VAD_MIN_DURATION_OFF`: Min silence to split (default: 0.0)

**Diarization Model**:
- `DIARIZATION_MODEL`: Model identifier (e.g., `pyannote/speaker-diarization-3.1`)
- `MIN_SPEAKERS`: Optional minimum speaker count
- `MAX_SPEAKERS`: Optional maximum speaker count

**Device Configuration**:
- `DEVICE`: Device string (`cpu`, `cuda`, `mps`) or auto-detect
- `COMPUTE_TYPE`: Compute type (`float16`, `int8`, `float32`) or auto-detect

**Authentication**:
- `HF_TOKEN`: Hugging Face authentication token (required for gated models)

### 5.2 Default Behaviors
- Device auto-detection: CUDA > MPS > CPU
- Compute type auto-detection: float16 (CUDA), int8 (CPU/MPS)
- MPS devices default to CPU for WhisperX compatibility (if needed)

---

## 6. Implementation Requirements

### 6.1 PyTorch Security (PyTorch 2.6+)

**Issue**: PyTorch 2.6+ restricts which objects can be loaded from model files.

**Requirement**: Must add safe globals allowlist before loading models:
```python
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    # ... other required types
])
```

**Required Types** (non-exhaustive):
- Omegaconf types (ListConfig, DictConfig, etc.)
- Pyannote core types (Introspection, Specifications, etc.)
- PyTorch Lightning types (EarlyStopping, ModelCheckpoint)
- Standard Python types (list, dict, set, int, float, str)

### 6.2 Chinese Language Support

**Character-Level Alignment**:
- Must detect Chinese languages: `zh`, `zh-cn`, `zh-tw`, `chinese`
- Must use character-level alignment for Chinese
- Must redistribute timestamps based on character position to prevent drift

**Timestamp Fixing Algorithm**:
1. Calculate character positions for each word in segment text
2. Redistribute timestamps proportionally based on character position
3. Ensure minimum word duration (e.g., 0.05 seconds)
4. Handle spaces correctly (Chinese has no spaces between words)

### 6.3 Memory Management

**Lazy Loading**:
- Diarization pipeline should be loaded on-demand (not at initialization)
- Pipeline should be deleted immediately after use

**Cleanup**:
- All components should provide cleanup methods
- ModelManager.cleanup() should call garbage collection
- CUDA cache should be cleared if available

**Resource Lifecycle**:
1. Initialize transcriber, aligner, embedder at startup
2. Load diarization pipeline on-demand per file
3. Delete diarization pipeline after each file
4. Run cleanup after processing

### 6.4 Error Handling

**Model Loading Errors**:
- Must handle missing Hugging Face token gracefully
- Must handle device incompatibilities (fallback to CPU)
- Must print clear error messages

**Processing Errors**:
- Should not crash the entire system
- Should log errors and continue processing
- Should handle corrupt model files gracefully

---

## 7. Default Implementations

### 7.1 WhisperXTranscriber

**Base Class**: `Transcriber`

**Model**: WhisperX (from GitHub: `m-bain/whisperx`)

**Features**:
- Integrated VAD (Voice Activity Detection)
- Configurable VAD parameters
- Support for multiple model sizes
- Language auto-detection

**Initialization Parameters**:
- `device`: Device string
- `compute_type`: Compute type string
- `vad_options`: Dictionary of VAD parameters
- `vad_method`: VAD method string

### 7.2 WhisperXAligner

**Base Class**: `Aligner`

**Model**: WhisperX alignment models (language-specific)

**Features**:
- Word-level alignment for most languages
- Character-level alignment for Chinese
- Timestamp drift fixing for Chinese
- Empty segment filtering

**Initialization Parameters**:
- `device`: Device string

### 7.3 PyannoteDiarizer

**Base Class**: `Diarizer`

**Model**: Pyannote speaker diarization pipeline

**Features**:
- Multi-speaker detection
- Optional speaker count constraints
- Lazy loading (on-demand)

**Initialization Parameters**:
- `device`: Device string (converted to torch.device)
- `model_name`: Model identifier string
- `hf_token`: Hugging Face token

**Methods**:
- `get_pipeline()`: Get or create pipeline (lazy loading)
- `cleanup()`: Delete pipeline and free memory

### 7.4 PyannoteEmbedder

**Base Class**: `Embedder`

**Model**: Pyannote embedding model (`pyannote/embedding`)

**Features**:
- Voice embedding generation
- Automatic tensor/array conversion
- Batch dimension handling

**Initialization Parameters**:
- `device`: Device string (converted to torch.device)
- `hf_token`: Hugging Face token

---

## 8. Integration Points

### 8.1 Pipeline Integration

**Transcription Stage**:
```python
result = model_manager.transcriber.transcribe(
    audio,
    batch_size=config.BATCH_SIZE,
    language=config.WHISPER_LANGUAGE,
    task="transcribe"
)
```

**Alignment Stage**:
```python
result = model_manager.align(
    result["segments"],
    audio,
    audio_sample_rate,
    result["language"]
)
```

**Diarization Stage**:
```python
diarization_segments = model_manager.diarizer.diarize(
    audio_dict,
    min_speakers=config.MIN_SPEAKERS,
    max_speakers=config.MAX_SPEAKERS
)
```

**Embedding Stage**:
```python
fingerprint = model_manager.embedder.embed(audio_dict)
```

### 8.2 Configuration Integration

Models read configuration from `config.py`:
- Device settings
- Model identifiers
- Processing parameters
- Authentication tokens

---

## 9. Extension Points

### 9.1 Adding New Model Implementations

To add a new model implementation:

1. **Create Implementation Class**:
   - Inherit from appropriate base class (`Transcriber`, `Aligner`, `Diarizer`, or `Embedder`)
   - Implement required methods with correct signatures
   - Follow input/output format specifications

2. **Register with ModelManager**:
   ```python
   custom_transcriber = CustomTranscriber(...)
   model_manager = ModelManager(transcriber=custom_transcriber)
   ```

3. **Ensure Compatibility**:
   - Input/output formats must match specifications
   - Device handling must be consistent
   - Error handling should be robust

### 9.2 Example: Custom Transcriber

```python
from models.base import Transcriber

class CustomTranscriber(Transcriber):
    def __init__(self, ...):
        # Initialize your model
        pass
    
    def transcribe(self, audio, batch_size=16, language=None, task="transcribe", **kwargs):
        # Your implementation
        return {
            "segments": [...],
            "language": "..."
        }
```

---

## 10. Testing Requirements

### 10.1 Component Testing

Each model component should be testable in isolation:
- Test with sample audio inputs
- Verify output format compliance
- Test error handling
- Test device compatibility

### 10.2 Integration Testing

Test ModelManager with:
- Default implementations
- Custom implementations
- Multiple device types
- Various audio formats
- Error scenarios

### 10.3 Performance Testing

- Measure model loading time
- Measure processing time per stage
- Monitor memory usage
- Test cleanup effectiveness

---

## 11. Known Constraints & Workarounds

### 11.1 PyTorch 2.6+ Security
- **Constraint**: Restricted object loading
- **Workaround**: Safe globals allowlist (see Section 6.1)

### 11.2 MPS Device Support
- **Constraint**: WhisperX may not fully support MPS
- **Workaround**: Default to CPU on Mac devices

### 11.3 Chinese Timestamp Drift
- **Constraint**: Character-level alignment can cause timestamp drift
- **Workaround**: Timestamp redistribution algorithm (see Section 6.2)

### 11.4 Memory Management
- **Constraint**: Large models consume significant memory
- **Workaround**: Lazy loading, explicit cleanup, garbage collection

---

## 12. Dependencies

### 12.1 Required Libraries
- **WhisperX**: `git+https://github.com/m-bain/whisperx.git`
- **Pyannote.Audio**: `>=3.4.0,<4.0.0`
- **PyTorch**: Latest stable (with CUDA support if using GPU)
- **NumPy**: For array operations
- **Pandas**: For DataFrame operations (diarization)

### 12.2 External Services
- **Hugging Face**: Required for model downloads
  - Models: `pyannote/embedding`, `pyannote/speaker-diarization-3.1`
  - Requires `HF_TOKEN` with read permissions
  - Must accept user conditions for gated models

---

## Appendix A: Quick Reference

### Model Component Summary
| Component | Purpose | Default Implementation | Key Feature |
|-----------|---------|------------------------|-------------|
| Transcriber | Audio → Text | WhisperXTranscriber | VAD, multi-language |
| Aligner | Refine timestamps | WhisperXAligner | Character-level for Chinese |
| Diarizer | Speaker separation | PyannoteDiarizer | Multi-speaker detection |
| Embedder | Voice embeddings | PyannoteEmbedder | Speaker identification |

### Key Constants
- **Sample Rate**: 16000 Hz (all audio processing)
- **Min Segment Duration**: 0.5 seconds (for reliable fingerprinting)
- **Default Batch Size**: 16
- **Default VAD Chunk Size**: 5 seconds

### Audio Format Summary
- **Transcription/Alignment**: NumPy array, shape `(samples,)`, 16kHz
- **Diarization/Embedding**: Dict with `{'waveform': tensor, 'sample_rate': 16000}`

---

**End of Models Architecture Specification**
