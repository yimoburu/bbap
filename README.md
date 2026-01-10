# Black Box Audio Protocol (BBAP) - System Specification

**Version:** 1.0  
**Last Updated:** 2025-01-09  
**Purpose:** Comprehensive specification for implementing an audio transcription, diarization, and speaker identity tracking system.

---

## 1. System Overview

### 1.1 Purpose
BBAP is a modular audio processing pipeline that:
- Transcribes long-form audio files (multi-speaker conversations)
- Identifies and separates different speakers (diarization)
- Tracks speaker identities across multiple files using voice fingerprinting
- Generates timestamped transcripts with speaker labels

### 1.2 Core Capabilities
1. **Transcription**: Convert speech to text with word-level timestamps
2. **Alignment**: Align transcribed text with audio at word/character level
3. **Diarization**: Identify "who spoke when" in multi-speaker audio
4. **Identity Tracking**: Match speakers across files using voice embeddings
5. **Output Generation**: Create formatted transcripts with absolute timestamps

### 1.3 Design Principles
- **Modularity**: Each processing stage is independent and replaceable
- **Incremental Processing**: Only processes new files (tracks processed files)
- **Self-Learning**: Automatically creates new speaker identities when unknown voices are detected
- **Format Agnostic**: Supports common audio formats (M4A, MP3, WAV, FLAC, AAC)

---

## 2. Functional Requirements

### 2.1 Input Requirements
- **Audio Files**: Must support formats readable by WhisperX (M4A, MP3, WAV, FLAC, AAC)
- **Filename Convention**: Optional timestamp format `yyyymmdd-hhmmss.ext` for absolute timestamp extraction
- **Input Location**: Configurable directory (default: `local_data/Voice Record Pro/`)

### 2.2 Processing Requirements
1. **Automatic Detection**: System must detect new, unprocessed audio files
2. **Sequential Processing**: Process files one at a time (no parallel processing requirement)
3. **Error Handling**: Continue processing remaining files if one fails
4. **Progress Tracking**: Log processed files to prevent reprocessing

### 2.3 Output Requirements
1. **Transcript Format**: Plain text files with structured format (see Section 8)
2. **Timestamp Format**: `[YYYY-MM-DD HH:MM:SS] Person_XXX: text`
3. **Speaker Labels**: Persistent IDs (e.g., `Person_001`, `Person_002`) that persist across files
4. **Daily Reports**: Generate summary reports of processing activity

### 2.4 Identity Management Requirements
1. **Vector Database**: Store voice embeddings as NumPy arrays (`.npy` files)
2. **Similarity Matching**: Use cosine similarity for voice comparison
3. **Threshold-Based Matching**: Configurable similarity threshold (default: 0.65)
4. **Automatic ID Generation**: Create new IDs (`Person_XXX`) for unmatched speakers
5. **Cross-File Consistency**: Same speaker in different files gets same ID

---

## 3. System Architecture Specification

### 3.1 Component Overview
The system consists of five main components:

```
┌─────────────┐
│   main.py   │  Entry point, file discovery, orchestration
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ pipeline.py │  Core processing logic (4-stage pipeline)
└──────┬──────┘
       │
       ├──► models.py ────► AI model management
       └──► utils.py  ────► Helper functions
```

### 3.2 Component Responsibilities

#### 3.2.1 `main.py` (Orchestrator)
- **Responsibilities**:
  - Scan input directory for new audio files
  - Track processed files (read/write log file)
  - Initialize ModelManager
  - Invoke pipeline for each new file
  - Generate daily reports
  - Handle SSL certificate issues (if needed)

- **Key Functions**:
  - File discovery (compare input directory against processed log)
  - Batch processing loop
  - Error aggregation and reporting

#### 3.2.2 `pipeline.py` (Core Pipeline)
- **Responsibilities**:
  - Execute 4-stage processing pipeline
  - Coordinate between transcription, alignment, diarization, and identification
  - Format and save output transcripts
  - Return list of newly discovered identities

- **Pipeline Stages** (executed sequentially):
  1. **Transcribe**: Generate text and segment-level timestamps
  2. **Align**: Refine to word/character-level timestamps
  3. **Diarize**: Assign speaker labels to segments
  4. **Identify**: Match speakers to known identities or create new ones
  5. **Format & Save**: Generate output transcript file

#### 3.2.3 `models/` (Model Architecture)
- **Responsibilities**:
  - Modular model component architecture
  - Unified interface for all AI model operations
  - Device configuration and memory management
  - Support for easy model swapping

- **Architecture**: See [`models/README.md`](models/README.md) for complete specification
  - **Transcriber**: Converts audio to text with segment-level timestamps
  - **Aligner**: Refines timestamps to word/character level
  - **Diarizer**: Identifies "who spoke when" in multi-speaker audio
  - **Embedder**: Generates voice embeddings for speaker identification

- **Default Implementations**:
  - **Transcription**: WhisperX
  - **Alignment**: WhisperX (with Chinese character-level support)
  - **Diarization**: Pyannote speaker diarization
  - **Embedding**: Pyannote embedding

- **Design**: Modular architecture allows easy swapping of model implementations

#### 3.2.4 `utils.py` (Utilities)
- **Responsibilities**:
  - File I/O operations
  - Timestamp parsing from filenames
  - Speaker ID generation and management
  - Vector database loading
  - Directory creation

- **Key Functions**:
  - `parse_timestamp(filename)`: Extract datetime from filename pattern
  - `load_known_speakers()`: Load all `.npy` files from vector database
  - `get_next_speaker_id(existing_names)`: Generate next available Person_XXX ID
  - `get_processed_files()`: Read processed log file
  - `mark_as_processed(filename)`: Append to processed log

#### 3.2.5 `config.py` (Configuration)
- **Responsibilities**:
  - Centralized configuration management
  - Environment variable loading
  - Path resolution (local vs cloud)
  - Device detection and configuration
  - Default value management

---

## 4. Processing Pipeline Specification

### 4.1 Pipeline Flow
```
Audio File
    │
    ▼
[1] TRANSCRIBE ──► WhisperX transcription with VAD
    │              Output: segments with text and timestamps
    ▼
[2] ALIGN ──────► Word/character-level alignment
    │              Output: refined segments with word timestamps
    ▼
[3] DIARIZE ────► Speaker separation
    │              Output: segments with speaker labels (SPEAKER_00, SPEAKER_01)
    ▼
[4] IDENTIFY ───► Voice fingerprinting and matching
    │              Output: segments with Person_XXX labels
    ▼
[5] FORMAT ─────► Generate transcript file
    │
    ▼
Output File
```

### 4.2 Stage 1: Transcription
**Input**: Audio file path  
**Output**: Dictionary with `segments` (list) and `language` (string)

**Specification**:
- Uses Transcriber component (default: WhisperX)
- Audio must be resampled to 16kHz mono
- VAD (Voice Activity Detection) performed by transcriber
- Supports configurable batch size and language settings

**Output Format**: See [`models/README.md`](models/README.md) Section 2.1 for complete specification

**Model Requirements**: See [`models/README.md`](models/README.md) for detailed interface and requirements

### 4.3 Stage 2: Alignment
**Input**: Segments from transcription, audio array, sample rate, language code  
**Output**: Segments with word-level timestamps

**Specification**:
- Uses Aligner component (default: WhisperX)
- For Chinese: Uses character-level alignment with timestamp fixing
- Handles empty segments and validation

**Output Format**: See [`models/README.md`](models/README.md) Section 2.2 for complete specification

**Critical Requirements**:
- Chinese languages require character-level alignment
- Timestamp drift must be prevented via redistribution algorithm
- See [`models/README.md`](models/README.md) Section 6.2 for implementation details

### 4.4 Stage 3: Diarization
**Input**: Audio as dict `{'waveform': torch.Tensor, 'sample_rate': int}`  
**Output**: Pandas DataFrame with speaker segments

**Specification**:
- Uses Diarizer component (default: Pyannote)
- Audio format: `{'waveform': tensor with shape (1, samples), 'sample_rate': 16000}`
- Supports optional `min_speakers` and `max_speakers` parameters

**Output Format**: See [`models/README.md`](models/README.md) Section 2.3 for complete specification

**Speaker Assignment**:
- Uses `whisperx.assign_word_speakers(diarization_segments, aligned_result)`
- Assigns speaker labels from diarization to word segments in transcription

### 4.5 Stage 4: Identity Resolution
**Input**: Segments with speaker labels (SPEAKER_00, SPEAKER_01, etc.)  
**Output**: Segments with Person_XXX labels, list of new identities

**Specification**:
- For each unique speaker in the file:
  1. Find longest continuous segment (better for fingerprinting)
  2. Skip if segment < 0.5 seconds
  3. Extract audio segment from in-memory waveform
  4. Generate embedding using Pyannote embedding model
  5. Compare against known speakers using cosine similarity
  6. If similarity > threshold: assign existing ID
  7. If similarity ≤ threshold: create new ID, save vector

**Embedding Generation**:
- Uses Embedder component (default: Pyannote)
- Audio format: `{'waveform': tensor shape (1, samples), 'sample_rate': 16000}`
- Output: 1D NumPy array (see [`models/README.md`](models/README.md) Section 2.4)

**Similarity Calculation**:
```
cosine_similarity = dot(a, b) / (norm(a) * norm(b))
```

**Decision Logic**:
- If `similarity > SIMILARITY_THRESHOLD` (default: 0.65): Match found
- Else: Create new identity

**Vector Storage**:
- Save as `.npy` file: `{VECTOR_DB_DIR}/{Person_XXX}.npy`
- Update in-memory cache for subsequent segments in same file

### 4.6 Stage 5: Format & Save
**Input**: Segments with Person_XXX labels  
**Output**: Text file

**Specification**:
- Extract start time from filename (format: `yyyymmdd-hhmmss.ext`)
- Calculate absolute timestamp for each segment: `start_time + segment['start']`
- Format: `[YYYY-MM-DD HH:MM:SS] Person_XXX: text`
- Fallback: If no timestamp in filename, use relative time

**Output Format**:
```
[2025-12-31 15:30:29] Person_010: 你跟老大的关系破裂也是替罪羊吗
[2025-12-31 15:30:33] Person_012: 那必然的呀
[2025-12-31 15:30:34] Person_012: 每个人都不愿意承认自己的错误然后都想找个替罪羊
```

---

## 5. Data Formats & Structures

### 5.1 Audio Input Format
- **Supported Formats**: M4A, MP3, WAV, FLAC, AAC
- **Processing**: Resampled to 16kHz mono by WhisperX
- **In-Memory Format**: NumPy array, shape `(samples,)` for mono audio

### 5.2 Pyannote Audio Format
When passing audio to Pyannote models (Diarizer and Embedder), use:
```python
{
    'waveform': torch.Tensor,  # Shape: (1, samples) for mono
    'sample_rate': int         # 16000
}
```

**See**: [`models/README.md`](models/README.md) Section 4.1 for complete audio format specifications

### 5.3 Vector Database Format
- **Storage**: NumPy arrays (`.npy` files)
- **Location**: `{PROJECT_ROOT}/Voice_Bank/Vectors/`
- **Naming**: `Person_XXX.npy` where XXX is zero-padded 3-digit number
- **Format**: 1D NumPy array (embedding vector)

### 5.4 Processed Log Format
- **Location**: `{PROJECT_ROOT}/processed_log.txt`
- **Format**: One filename per line (plain text)
- **Purpose**: Track which files have been processed

### 5.5 Daily Report Format
- **Location**: `{PROJECT_ROOT}/Daily_Reports/Briefing_{YYYY-MM-DD}.md`
- **Format**: Markdown
- **Content**:
  - Processing stats (files processed, errors)
  - New identities created (list of Person_XXX IDs)

---

## 6. Configuration Specification

### 6.1 Environment Variables
All configuration via `.env` file or environment variables:

#### 6.1.1 Required
- `HF_TOKEN`: Hugging Face authentication token (read permissions required)

#### 6.1.2 Path Configuration
- `LOCAL_MODE`: `True`/`False` - Use local data directory vs Google Drive
- `BASE_DIR`: Base directory path (default: `local_data` if LOCAL_MODE, else Google Drive)
- `PROJECT_ROOT_NAME`: Project folder name (default: `yuzhe`)
- `INPUT_FOLDER_NAME`: Input audio folder name (default: `Voice Record Pro`)

#### 6.1.3 Model Configuration
- `WHISPER_MODEL`: Model size - `tiny`, `base`, `small`, `medium`, `large-v3` (default: `large-v3`)
- `WHISPER_LANGUAGE`: Language code or `None` for auto-detect (e.g., `zh`, `en`)
- `DIARIZATION_MODEL`: `pyannote/speaker-diarization-3.1` or `pyannote/speaker-diarization-3.0`

**See**: [`models/README.md`](models/README.md) Section 5 for complete model configuration requirements

#### 6.1.4 Processing Parameters
- `DEVICE`: `cpu`, `cuda`, or `mps` (auto-detected if not set)
- `COMPUTE_TYPE`: `float16` (GPU), `int8` (CPU), or `float32`
- `BATCH_SIZE`: Batch size for transcription (default: 16)
- `SIMILARITY_THRESHOLD`: Cosine similarity threshold for identity matching (default: 0.65)
- `MIN_SPEAKERS`: Minimum speaker count (optional, default: 2)
- `MAX_SPEAKERS`: Maximum speaker count (optional, default: 10)
- `DIARIZATION_CLUSTERING_THRESHOLD`: Diarization sensitivity (default: 0.7)

#### 6.1.5 WhisperX VAD Configuration
- `WHISPERX_VAD_METHOD`: `pyannote` (default, accurate) or `silero` (faster)
- `WHISPERX_VAD_ONSET`: Speech start threshold 0.0-1.0 (default: 0.5)
- `WHISPERX_VAD_OFFSET`: Speech end threshold 0.0-1.0 (default: 0.363)
- `WHISPERX_VAD_CHUNK_SIZE`: Max chunk size in seconds (default: 5, original: 30)
- `WHISPERX_VAD_MIN_DURATION_ON`: Min speech duration in seconds (default: 0.0)
- `WHISPERX_VAD_MIN_DURATION_OFF`: Min silence to split in seconds (default: 0.0)

### 6.2 Default Behavior
- Device auto-detection: CUDA > MPS > CPU
- Compute type auto-detection: float16 (CUDA), int8 (CPU/MPS)
- MPS devices default to CPU for WhisperX compatibility

---

## 7. Component Interface Specifications

### 7.1 ModelManager Interface

**See**: [`models/README.md`](models/README.md) Section 3 for complete ModelManager specification

**Key Methods**:
- `ModelManager.__init__(transcriber=None, aligner=None, diarizer=None, embedder=None)`: Initialize with model components
- `ModelManager.align(segments, audio, audio_sample_rate, language_code)`: Align transcription segments
- `ModelManager.get_diarization_pipeline()`: Get diarization pipeline
- `ModelManager.cleanup()`: Free memory and run garbage collection

**Architecture**: Modular design allows custom model implementations. See [`models/README.md`](models/README.md) for details.

### 7.2 Pipeline Interface

#### `pipeline.run(file_path, filename, model_manager)`
- **Purpose**: Process single audio file through full pipeline
- **Parameters**:
  - `file_path`: Full path to audio file
  - `filename`: Base filename (for output naming)
  - `model_manager`: ModelManager instance
- **Returns**: List of newly created Person IDs (strings)
- **Side Effects**: 
  - Creates transcript file in OUTPUT_DIR
  - Creates/updates vector database files
  - Updates in-memory speaker cache

### 7.3 Utility Functions

#### `utils.parse_timestamp(filename)`
- **Purpose**: Extract datetime from filename
- **Input**: Filename string (e.g., `20251231-153029.m4a`)
- **Output**: `datetime` object or `None` if pattern not found
- **Pattern**: `yyyymmdd-hhmmss` (8 digits, dash, 6 digits)

#### `utils.load_known_speakers()`
- **Purpose**: Load all speaker vectors from vector database
- **Output**: Dictionary `{Person_XXX: numpy_array, ...}`
- **Error Handling**: Skips corrupt files with warning

#### `utils.get_next_speaker_id(existing_names)`
- **Purpose**: Generate next available Person_XXX ID
- **Input**: Set or list of existing Person IDs
- **Output**: String like `Person_013` (zero-padded, increments from highest existing)

---

## 8. Output Format Specification

### 8.1 Transcript File Format
- **File Extension**: `.txt`
- **Encoding**: UTF-8
- **Location**: `{PROJECT_ROOT}/Processed_Conversations/{basename}.txt`
- **Line Format**: `[YYYY-MM-DD HH:MM:SS] Person_XXX: text`

### 8.2 Timestamp Format
- **Format**: `YYYY-MM-DD HH:MM:SS` (ISO 8601-like, space-separated)
- **Source**: Extracted from filename pattern `yyyymmdd-hhmmss.ext` + segment offset
- **Calculation**: `filename_start_time + segment['start']` (in seconds)
- **Fallback**: If no timestamp in filename, use relative time `HH:MM:SS` (from start of file)

### 8.3 Speaker Label Format
- **Format**: `Person_XXX` where XXX is zero-padded 3-digit number (001, 002, ..., 010, ...)
- **Special Case**: `Unknown` if speaker could not be determined
- **Persistence**: Same speaker across files gets same Person_XXX ID

### 8.4 Example Output
```
[2025-12-31 15:30:29] Person_010: 
[2025-12-31 15:30:33] Person_012: 
[2025-12-31 15:30:34] Person_012: 
```

---

## 9. Error Handling & Edge Cases

### 9.1 File Processing Errors
- **Behavior**: Continue processing remaining files if one fails
- **Logging**: Print error message, continue to next file
- **Tracking**: Failed files are NOT added to processed log (will retry on next run)

### 9.2 Audio Format Errors
- **Prevention**: Use in-memory audio slicing (don't re-open files)
- **Format Support**: Rely on WhisperX's format support (M4A, MP3, WAV, FLAC, AAC)

### 9.3 Model Loading Errors
- **Hugging Face Auth**: Must accept user conditions for gated models
- **Missing Token**: Print warning, may fail during model loading
- **Device Issues**: Auto-fallback (MPS → CPU for WhisperX)

### 9.4 Identity Matching Edge Cases
- **Short Segments**: Skip segments < 0.5 seconds (unreliable for fingerprinting)
- **No Match**: Create new identity if similarity ≤ threshold
- **Multiple Matches**: Use highest similarity score
- **Vector Corruption**: Skip corrupt `.npy` files with warning

### 9.5 Timestamp Edge Cases
- **No Filename Timestamp**: Use relative time (from file start)
- **Invalid Timestamp Format**: Fall back to relative time
- **Timestamp Drift (Chinese)**: Fixed by redistribution algorithm in alignment stage

### 9.6 Known Issues & Workarounds

#### SSL Certificate Errors
- **Issue**: Python SSL certificate verification failures
- **Solution**: Use `certifi` library to patch SSL context (implemented in `main.py`)

#### PyTorch 2.6+ Security Restrictions
- **Issue**: `WeightsUnpickler error: Unsupported global: GLOBAL omegaconf...`
- **Solution**: Add safe globals allowlist using `torch.serialization.add_safe_globals()` (implemented in `models.py`)

#### WhisperX Single Segment Issue
- **Issue**: Transcription returns single segment for multi-speaker audio
- **Solution**: Adjust VAD parameters:
  - Lower `WHISPERX_VAD_CHUNK_SIZE` (e.g., 5 instead of 30)
  - Lower `WHISPERX_VAD_ONSET` and `WHISPERX_VAD_OFFSET` for more sensitivity

#### Chinese Character Timestamp Alignment
- **Issue**: Timestamps drift for Chinese characters
- **Solution**: Character-level alignment + timestamp redistribution based on character position

---

## 10. Dependencies & Technical Requirements

### 10.1 Core Dependencies
- **WhisperX**: `git+https://github.com/m-bain/whisperx.git` (must be from Git, not PyPI)
- **Pyannote.Audio**: `>=3.4.0,<4.0.0` (version constraint required)
- **PyTorch**: Latest stable (with CUDA support if using GPU)
- **NumPy**: For array operations and vector storage
- **Pandas**: For DataFrame operations (diarization segment conversion)
- **Python-dotenv**: For environment variable management

### 10.2 System Requirements
- **Python**: 3.8+ (tested on 3.13)
- **FFmpeg**: Required for audio format support (system dependency)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
- **Storage**: ~250GB per year for 12 hours/day recording (at ~100-130 kbps)

### 10.3 Hardware Requirements
- **CPU**: Any modern CPU (processing will be slow)
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster processing)
- **Mac**: M1/M2/M3 with MPS support (falls back to CPU for WhisperX compatibility)

### 10.4 External Services
- **Hugging Face**: Required for model downloads
  - Must accept user conditions for:
    - `pyannote/embedding`
    - `pyannote/speaker-diarization-3.1` (or 3.0)
  - Requires `HF_TOKEN` with read permissions

---

## 11. Implementation Notes

### 11.1 Critical Implementation Details

#### Audio Format Handling
- **Always use in-memory slicing**: Extract audio segments from loaded waveform array
- **Never re-open files**: Avoids format errors with M4A/MP3 files
- **Pyannote format**: Always pass audio as `{'waveform': tensor, 'sample_rate': int}`

#### Device Management
- **String to Device**: Convert `config.DEVICE` (string) to `torch.device` for Pyannote
- **MPS Fallback**: WhisperX doesn't fully support MPS, default to CPU on Mac
- **Memory Cleanup**: Delete diarization pipeline after use, run garbage collection

#### Chinese Language Support
- **Character-Level Alignment**: Required for accurate Chinese timestamps
- **Timestamp Fixing**: Redistribute timestamps within segments to prevent drift
- **Implementation**: Check language code, use `return_char_alignments=True` for Chinese

#### Vector Database Management
- **In-Memory Cache**: Load all vectors at start of file processing
- **Update Cache**: Add new identities to cache immediately (for same-file matching)
- **File Naming**: Zero-padded 3-digit format (`Person_001`, not `Person_1`)

### 11.2 Performance Optimizations
- **Batch Processing**: Use configurable batch size for transcription
- **Selective Model Loading**: Load diarization pipeline on-demand
- **Memory Management**: Delete models after use, explicit garbage collection
- **In-Memory Audio**: Keep audio in memory throughout pipeline (avoid re-loading)

### 11.3 Security Considerations
- **Environment Variables**: Store sensitive tokens in `.env` file (not in code)
- **Safe Globals**: PyTorch 2.6+ requires explicit allowlist for loaded objects
- **SSL Certificates**: Use `certifi` for reliable certificate handling

---

## 12. Testing & Validation

### 12.1 Test Scenarios
1. **Single Speaker Audio**: Verify correct transcription and identity assignment
2. **Multi-Speaker Audio**: Verify diarization separates speakers correctly
3. **Cross-File Identity**: Process multiple files with same speaker, verify same ID
4. **New Speaker**: Verify new identity creation when similarity < threshold
5. **Chinese Audio**: Verify character-level alignment and timestamp accuracy
6. **Missing Timestamp**: Verify fallback to relative time
7. **Error Recovery**: Verify system continues after file processing error

### 12.2 Validation Criteria
- **Accuracy**: Transcription should match audio content
- **Diarization**: Speakers should be correctly separated
- **Identity Consistency**: Same speaker across files gets same Person_XXX ID
- **Timestamp Accuracy**: Timestamps should align with audio playback
- **Output Format**: Transcripts must match specified format exactly

---

## 13. Future Enhancements (Out of Scope)

The following features are not part of the current specification but may be considered for future versions:
- Parallel file processing
- Real-time processing
- Web interface
- Manual identity merging tools
- Advanced clustering algorithms
- Multi-language support improvements
- Cloud storage integration (beyond Google Drive)

---

## Appendix A: File Structure

```
bbap/
├── main.py              # Entry point, orchestration
├── pipeline.py          # Core processing pipeline
├── models.py            # Model management
├── utils.py             # Utility functions
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (user-created)
├── README.md            # This specification
└── local_data/          # Data directory (if LOCAL_MODE=True)
    ├── Voice Record Pro/        # Input audio files
    └── yuzhe/                   # Project root
        ├── Processed_Conversations/  # Output transcripts
        ├── Voice_Bank/
        │   └── Vectors/          # Speaker embeddings (.npy files)
        ├── Daily_Reports/       # Daily briefing reports
        └── processed_log.txt    # Processed files log
```

---

## Appendix B: Quick Reference

### Key Constants
- **Sample Rate**: 16000 Hz (all audio processing)
- **Similarity Threshold**: 0.65 (cosine similarity for identity matching)
- **Min Segment Duration**: 0.5 seconds (for reliable fingerprinting)
- **Default Batch Size**: 16
- **Default VAD Chunk Size**: 5 seconds

### Key Models
- **Transcription**: WhisperX (configurable model size)
- **Embedding**: `pyannote/embedding`
- **Diarization**: `pyannote/speaker-diarization-3.1`

### Output Locations
- **Transcripts**: `{PROJECT_ROOT}/Processed_Conversations/`
- **Vectors**: `{PROJECT_ROOT}/Voice_Bank/Vectors/`
- **Reports**: `{PROJECT_ROOT}/Daily_Reports/`
- **Log**: `{PROJECT_ROOT}/processed_log.txt`

---

**End of Specification**
