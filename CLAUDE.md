# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stego-aas** is a Django REST Framework web application implementing SparSamp, a steganographic protocol that hides messages within LLM-generated text in a provably secure way. The application provides API endpoints for encoding plaintext into stego-text and decoding it back.

### Key Innovation: BackCheck Algorithm

The codebase implements a BackCheck algorithm to solve token ambiguity during decoding. This is the core technical challenge: the same text can be tokenized in multiple valid ways, and finding the correct tokenization that was used during encoding is critical for successful message recovery.

## Architecture

### Django Project Structure

- **Project root**: `stego-aas/stegoaas/`
- **Django app**: `sparsamp_app/` (contains all steganography logic)
- **Django settings**: `stegoaas/settings.py`
  - Uses SQLite database
  - Includes custom setting: `SPARSAMP_CONTEXT_STRING`
- **URL routing**:
  - Main URLs in `stegoaas/urls.py`
  - App URLs in `sparsamp_app/urls.py`

### Core Components

#### 1. API Layer (`views.py`, `serializers.py`, `urls.py`)

Two REST endpoints under `/sparsamp_app/api/`:
- `POST /encode/` - Encodes plaintext into LLM-generated stego messages
- `POST /decode/` - Decodes stego messages back to plaintext

Request/response formats defined in `serializers.py`:
- EncodeRequestSerializer: `plaintext`, `context`, `random_seed`
- DecodeRequestSerializer: `messages` (list), `context`, `random_seed`
- Decode response includes `attempts_per_message` array with BackCheck performance metrics

#### 2. Encoding Pipeline (`encoding.py`)

Flow: `full_encode()` → `encode_spar()` → `encode_step()`

Key concepts:
- **Block-based encoding**: Messages are split into 32-bit blocks
- **Checkpoint mechanism**: Every 4th block reserves last 8 bits for verification (all zeros)
- **Sparse sampling**: Uses probability distributions from GPT-2 model to select tokens that encode message bits
- **State variables**: Maintains `n_m` (range) and `k_m` (encoded value) through encoding process
- Uses `process_message_with_checkpoints()` to prepare message bits with embedded checkpoints

#### 3. Decoding Pipeline (`decoding.py`)

Flow: `full_decode()` → `backcheck_decode_single_message()` → `BackCheckDecoder.backcheck_decode_tree()` → `try_blind_decoding()`

**Performance Tracking**:
- `full_decode()` returns tuple: `(decoded_message, attempts_list)`
- `attempts_list` contains BackCheck attempts needed for each message segment
- Useful for analyzing decoding efficiency and checkpoint effectiveness

**BackCheck Algorithm** (`backcheck.py`):
- Builds a tree of possible tokenizations using `BackCheckTree`
- Each node (`BackCheckNode`) represents a token choice with attributes:
  - `greedy_correct`: Part of default tokenization
  - `likeliness`: Probability from model
  - `known_correct`/`known_wrong`: Verification results
- **Path selection priority** (`_find_path_through_tree()`):
  1. Known correct paths
  2. Greedy (default tokenization) paths
  3. Highest probability paths
- Verifies paths using `try_blind_decoding()` which attempts to extract message bits
- Updates tree based on verification results, marks wrong paths, backtracks if needed
- Checkpoint verification: Confirms every 4th decoded block ends with `00000000`

**BlindDecodingState** (`decoding.py`):
- Tracks decoding progress: discovered bits, random state, model past states
- Maintains checkpoint counter (`backcheck_count`)
- Can save/restore random state for backtracking
- Converts decoded bits to UTF-8 string

#### 4. Token Graph (`token_graph.py`)

Helper module for tokenization analysis:
- `build_token_graph()`: Creates directed graph of all possible tokenizations for a given text
- `all_tokenizations()`: Generates all valid token sequences
- `best_next_tokenization()`: Uses Viterbi algorithm with model probabilities to find optimal tokenization given constraints

Note: Token graph is built during tests but BackCheck algorithm is the primary decoding approach.

#### 5. Model Management (`model_manager.py`)

Thread-safe singleton for GPT-2 model lifecycle:
- `ModelManager`: Singleton class managing model, tokenizer, and device
- `get_model_manager()`: Returns singleton instance
- Automatic GPU detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Lazy loading: Model loaded on first access
- Thread-safe initialization using lock

#### 6. Utilities (`sparsamp_utils.py`)

Shared utilities adapted from SparSamp research (CC BY 4.0):
- `get_probs_past()`: Gets token probability distribution from model with top-p sampling
- `string_to_utf8_binary()` / `utf8_binary_to_string()`: Message encoding
- `dec2bin()`: Integer to binary string conversion
- `get_lower_upper_bound()`: Calculates probability intervals for arithmetic coding
- `func_mrn()`: Maps encoded value to probability range

## Development Commands

### Setup and Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix

# Model is loaded automatically on first API request via ModelManager singleton
```

### Running the Application

```bash
cd stego-aas/stegoaas

# Run development server
python manage.py runserver

# Run database migrations
python manage.py migrate

# Create superuser for admin
python manage.py createsuperuser
```

### Testing

```bash
cd stego-aas/stegoaas

# Run all tests
python manage.py test

# Run specific test class
python manage.py test sparsamp_app.tests.test_sparsamp.SparSampTest

# Run single test method
python manage.py test sparsamp_app.tests.test_sparsamp.SparSampTest.test_short_text
```

Test files location: `sparsamp_app/tests/`
- `test_sparsamp.py`: Integration tests for encode/decode pipeline
- `test_views.py`: API endpoint tests for encode/decode views
- `test_token_graph.py`: Token graph utilities tests

### Django Management

```bash
# Django shell for testing
python manage.py shell

# Check for issues
python manage.py check

# View database schema
python manage.py dbshell
```

## Important Implementation Details

### Random Seed Consistency

The same `random_seed` MUST be used for encoding and decoding. The random number generator state is critical for:
1. Token selection during encoding
2. Verification during decoding
3. Sub-messages use derived seeds: `rng.integers(low=10**15, high=10**16)`

### Model Configuration

- Uses `openai-community/gpt2` model from HuggingFace
- Top-p sampling: 0.95 (filters low-probability tokens)
- Context truncation: Last 1022 tokens via `limit_past()`
- Device: Auto-detected via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - Automatically uses GPU when CUDA is available
  - Falls back to CPU if CUDA is not detected
  - No code changes needed—GPU used automatically when detected

### Checkpoint System

Critical for enabling backtracking:
- Inserted every 4 blocks (configurable via `blocks_per_interval`)
- Checkpoint blocks: 24 message bits + 8 zero bits
- Regular blocks: 32 message bits
- During decoding, checkpoint verification failure triggers backtracking in tree

### Token Length Constraints

Encoding parameters:
- `min_token_length`: 100 tokens (ensures enough tokens to embed message block)
- `max_token_length`: 1000 tokens (prevents infinite loops)
- Block size: 32 bits (fixed)

### Error Handling

Key failure modes:
- Token not found in probability distribution → decoding fails, backtrack
- `n_m <= 0` during decoding → invalid path, backtrack
- Checkpoint verification fails → wrong tokenization path, backtrack
- No valid path found after max attempts (500) → raises error

## Recent Changes and Refactorings

### Performance Stats Tracking (Current Branch)

This branch adds performance metrics tracking to the decoding pipeline:

**Changes Made**:
- `decoding.py`: Modified `full_decode()` to return `(decoded_message, attempts_list)` tuple
- `views.py`: Updated `SparsampDecodeView` to include `attempts_per_message` in response
- `backcheck.py`: `backcheck_decode_single_message()` returns attempts count
- API response now includes BackCheck performance data for analysis

**Benefits**:
- Track decoding efficiency across different messages
- Analyze checkpoint effectiveness
- Identify performance bottlenecks
- Monitor BackCheck algorithm convergence

### Code Refactorings

The codebase has been refactored to improve maintainability and code quality. These changes maintain full backward compatibility and all tests pass.

### 1. Centralized Configuration (`constants.py`)

All magic numbers and configuration values have been extracted into a dedicated constants module:

```python
# Encoding/Decoding Parameters
BLOCK_SIZE = 32
BLOCKS_PER_INTERVAL = 4
CHECKPOINT_MARKER_SIZE = 8

# Model Parameters
TOP_P = 0.95
CONTEXT_WINDOW_SIZE = 1022
MIN_TOKEN_LENGTH = 100
MAX_TOKEN_LENGTH = 1000

# Model Configuration
MODEL_NAME = "openai-community/gpt2"

# Random Seed Ranges
RANDOM_SEED_MIN = 10 ** 15
RANDOM_SEED_MAX = 10 ** 16

# BackCheck Parameters
MAX_BACKCHECK_ATTEMPTS = 10
```

Files updated to use constants:
- `encoding.py`
- `decoding.py`
- `sparsamp_utils.py`

### 2. Code Cleanup

Removed unused functions from `sparsamp_utils.py`:
- `bits2int`, `int2bits` - Unused bit conversion utilities
- `tokenize_first_n_words` - Alternative tokenization approach not in use
- `num_same_from_beg` - Unused comparison utility
- `get_logits`, `process_logits_to_probs` - Alternative probability calculation methods
- `find_nearest` - Unused binary search utility
- `load_context` - File loading utility not used in current implementation
- `get_bits_length_from_list` - Unused statistics function
- `custom_round` - Redundant rounding function

Removed unused imports:
- `typing.List` - No longer needed
- `re` - Regular expressions not used
- `math.ceil`, `math.floor` - Not used in simplified code

### 3. Improved Documentation

All functions now have comprehensive docstrings following Python conventions:

- **sparsamp_utils.py**: Added detailed docstrings for all utility functions, translated Chinese comments to English
- **encoding.py**: Documented `full_encode()` and `process_message_with_checkpoints()` with parameter descriptions and return types
- Clear explanations of checkpoint mechanism, binary encoding, and model configuration

### 4. Testing Verification

All refactorings were tested incrementally:
- `test_short_text`: Passes in 1 attempt consistently
- `test_middle_text`: All messages decode successfully in 1 attempt
- `test_multiple_texts`: BackCheck implementation verified with <10 attempts (typically 1-6 attempts)

These refactorings make the codebase easier to understand and maintain while preserving all functionality.

## Working with the Code

### Adding New API Endpoints

1. Define serializer in `serializers.py`
2. Create view class in `views.py` (inherit from `APIView`)
3. Add URL pattern in `sparsamp_app/urls.py`

### Modifying Encoding/Decoding Logic

- Encoding changes: Modify `encoding.py`, ensure tests pass
- Decoding changes: Modify `decoding.py`, update BackCheck logic carefully
- Keep checkpoint logic synchronized between encoder and decoder

### Testing Changes

Always test with multiple message lengths:
- Short messages (< 100 chars): `test_short_text`
- Medium messages (200-500 chars): `test_middle_text`
- Long messages (> 1000 chars): `test_long_text`

Test with various contexts and seeds to ensure robustness.

For API changes, run the view tests:
```bash
python manage.py test sparsamp_app.tests.test_views
```

### Performance Considerations

- BackCheck tree exploration can be expensive (max attempts configurable via `MAX_BACKCHECK_ATTEMPTS`)
- Model inference benefits from GPU acceleration when available
- Token graph generation scales with text length and vocabulary size
- Checkpoint frequency trades off message capacity vs. backtrack efficiency
- Performance metrics are tracked and returned via `attempts_per_message` in decode responses
- Typical BackCheck attempts: 1-6 for successful decoding

## Key Files Reference

- `sparsamp_app/views.py:13-38` - API endpoints with performance tracking
- `sparsamp_app/encoding.py:14-36` - Main encoding entry point
- `sparsamp_app/decoding.py:178-207` - Main decoding entry point with attempts tracking
- `sparsamp_app/backcheck.py:375-416` - BackCheck decode single message
- `sparsamp_app/backcheck.py:336-372` - BackCheck tree algorithm
- `sparsamp_app/backcheck.py:72-213` - BackCheckTree class
- `sparsamp_app/constants.py` - Configuration parameters
- `sparsamp_app/model_manager.py` - Singleton model management with GPU support
- `stegoaas/settings.py` - Django settings including SPARSAMP_CONTEXT_STRING
- Always run `test_short_text` after changes to verify integrity
- Run `test_multiple_texts` after bigger changes for comprehensive validation
- Integration tests located in `sparsamp_app/tests/test_sparsamp.py`
- API endpoint tests located in `sparsamp_app/tests/test_views.py`