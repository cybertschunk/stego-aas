from __future__ import annotations

import logging
import random
from math import ceil
from typing import List, Tuple

import numpy as np
import torch

from .constants import (
    BLOCK_SIZE, BLOCKS_PER_INTERVAL, MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH,
    RANDOM_SEED_MIN, RANDOM_SEED_MAX, TOP_P, CONTEXT_WINDOW_SIZE
)
from .model_manager import get_model_manager
from .sparsamp_utils import func_mrn, get_lower_upper_bound, get_probs_past, string_to_utf8_binary

logger = logging.getLogger(__name__)



def full_encode(context: str, message_text: str, random_seed: int) -> List[str]:
    """
    Encode a message into steganographic text using SparSamp algorithm.

    Args:
        context: Initial text context for generation
        message_text: Message to hide in the generated text
        random_seed: Random seed for reproducible encoding

    Returns:
        List of generated text messages containing the hidden information
    """
    model_manager = get_model_manager()
    message_bits = process_message_with_checkpoints(message_text, BLOCK_SIZE)
    logger.debug(f"Message bits length: {len(message_bits)}")

    final_messages = []
    tokenized_context = model_manager.tokenizer.encode(context, return_tensors='pt').to(model_manager.device)
    rng = np.random.default_rng(random_seed)

    bits_encoded = 0
    while bits_encoded < len(message_bits):
        remaining_bits = message_bits[bits_encoded:]
        random_number = rng.integers(low=RANDOM_SEED_MIN, high=RANDOM_SEED_MAX)

        generated_ids, encoded_messages = encode_spar(
            model=model_manager.model,
            context=tokenized_context,
            message_bits=remaining_bits,
            min_token_length=MIN_TOKEN_LENGTH,
            max_token_length=MAX_TOKEN_LENGTH,
            random_seed=random_number,
            device=model_manager.device
        )

        encoded_message = "".join(encoded_messages)
        decoded_text = model_manager.tokenizer.decode(generated_ids)
        final_messages.append(decoded_text)
        bits_encoded += len(encoded_message)

    logger.info(f"Encoded {bits_encoded} bits into {len(final_messages)} message(s)")
    return final_messages


def encode_step(probs: torch.Tensor, n_m: int, k_m: int) -> Tuple[int, int, int]:
    """
    Perform one step of SparSamp encoding.

    Args:
        probs: Probability distribution over tokens
        n_m: Current interval size
        k_m: Current message state

    Returns:
        Tuple of (token_index, new_n_m, new_k_m)
    """
    r = random.random()
    cumulative_probs = probs.cumsum(0)
    r_i_m = func_mrn(k_m, n_m, r)
    token_index = (cumulative_probs > r_i_m).nonzero()[0].item()

    lower, upper = get_lower_upper_bound(cumulative_probs, token_index)
    temp0 = ceil((lower - r) * n_m)
    temp1 = ceil((upper - r) * n_m)

    k_m = k_m - n_m - temp0 if k_m + r * n_m >= n_m else k_m - temp0
    n_m = temp1 - temp0

    return token_index, n_m, k_m


@torch.no_grad()
def encode_spar(
    model: torch.nn.Module,
    context: torch.Tensor,
    message_bits: str,
    min_token_length: int,
    max_token_length: int,
    device: str = 'cpu',
    block_size: int = BLOCK_SIZE,
    top_p: float = TOP_P,
    random_seed: int = 42
) -> Tuple[List[int], List[str]]:
    """
    Encode message bits into token sequence using SparSamp algorithm.

    Args:
        model: Language model
        context: Tokenized context
        message_bits: Binary string to encode
        min_token_length: Minimum tokens to generate per block
        max_token_length: Maximum tokens (safety limit)
        device: Device to run on
        block_size: Size of message blocks in bits
        top_p: Nucleus sampling parameter
        random_seed: Random seed

    Returns:
        Tuple of (generated token IDs, encoded message blocks)

    Raises:
        RuntimeError: If encoding exceeds maximum token length
    """
    context = context[-CONTEXT_WINDOW_SIZE:].clone().detach().to(device=device, dtype=torch.long)
    random.seed(random_seed)

    generated_ids = []
    encoded_message = []
    m_index = 0
    k_m = int(message_bits[:block_size], 2)
    n_m = 2 ** block_size
    token_count = 0
    past = None
    prev = context

    while True:
        probs, indices, past = get_probs_past(
            model=model,
            prev=prev,
            past=past,
            device=device,
            top_p=top_p
        )

        probs = probs.to(torch.float64)
        token_index, n_m, k_m = encode_step(probs, n_m, k_m)
        token_id = indices[token_index]
        token_count += 1

        can_encode_more = (m_index + block_size < len(message_bits))
        below_min_length = (token_count < min_token_length)

        if n_m == 1:
            encoded_message.append(message_bits[m_index:m_index + block_size])
            m_index += block_size

            if not (below_min_length and can_encode_more):
                generated_ids.append(token_id.item())
                break

            n_m = 2 ** block_size
            k_m = int(message_bits[m_index:m_index + block_size], 2)

        if token_count > max_token_length:
            raise RuntimeError(
                f"Exceeded maximum token length ({max_token_length}). "
                f"Context may be inappropriate for encoding."
            )

        generated_ids.append(token_id.item())
        prev = torch.tensor([[token_id.item()]], device=device, dtype=torch.long)

    logger.debug(f"Encoded message block: {len(generated_ids)} tokens, {len(encoded_message)} blocks")
    return generated_ids, encoded_message

def process_message_with_checkpoints(
    message_text: str,
    block_size: int = BLOCK_SIZE,
    blocks_per_interval: int = BLOCKS_PER_INTERVAL
) -> str:
    """
    Convert message to binary string with checkpoint markers.

    Every 'blocks_per_interval' blocks, a checkpoint is inserted containing
    (block_size - 8) message bits followed by 8 zero bits as a marker.

    Args:
        message_text: Message string to convert to binary
        block_size: Size of each block in bits
        blocks_per_interval: Number of blocks before checkpoint

    Returns:
        Binary string with checkpoint markers inserted
    """
    message_bits = string_to_utf8_binary(message_text)
    blocks = []
    idx = 0
    interval_counter = 0

    while idx < len(message_bits):
        interval_counter += 1

        if interval_counter == blocks_per_interval:
            take_len = block_size - 8
            block_bits = message_bits[idx:idx + take_len].ljust(take_len, '0')
            block_bits += "00000000"
            blocks.append(block_bits)
            idx += take_len
            interval_counter = 0
        else:
            block_bits = message_bits[idx:idx + block_size].ljust(block_size, '0')
            blocks.append(block_bits)
            idx += block_size

    return ''.join(blocks)
