"""
Constants for SparSamp steganography implementation.
"""

# Encoding/Decoding Parameters
BLOCK_SIZE = 32  # Size of message blocks in bits
BLOCKS_PER_INTERVAL = 4  # Blocks before checkpoint
CHECKPOINT_MARKER_SIZE = 8  # Checkpoint marker size in bits

# Model Parameters
TOP_P = 0.95  # Top-p (nucleus) sampling
CONTEXT_WINDOW_SIZE = 1022  # Max context size for model
MIN_TOKEN_LENGTH = 100  # Min tokens per message block
MAX_TOKEN_LENGTH = 1000  # Max tokens (safety limit)

# Model Configuration
MODEL_NAME = "openai-community/gpt2"

# Random Seed Ranges
RANDOM_SEED_MIN = 10 ** 15
RANDOM_SEED_MAX = 10 ** 16

# BackCheck Parameters
MAX_BACKCHECK_ATTEMPTS = 20  # Max attempts for BackCheck
