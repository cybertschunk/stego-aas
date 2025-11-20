from __future__ import annotations  # no effect in 3.8 but safe

import random
from math import ceil

import numpy as np
import torch

from .constants import (
    BLOCK_SIZE, BLOCKS_PER_INTERVAL, MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH,
    RANDOM_SEED_MIN, RANDOM_SEED_MAX, TOP_P, CONTEXT_WINDOW_SIZE
)
from .model_manager import get_model_manager
from .sparsamp_utils import func_mrn, get_lower_upper_bound, get_probs_past, string_to_utf8_binary



def full_encode(context, message_text, random_seed):
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
    print(message_bits)
    final_messages = []
    i = 0
    tokenized_context = model_manager.tokenizer.encode(context, return_tensors='pt').to(model_manager.device)
    rng = np.random.default_rng(random_seed)
    to_decode = message_bits
    all_generated_ids = []
    while i < len(message_bits):
        random_number = rng.integers(low=RANDOM_SEED_MIN, high=RANDOM_SEED_MAX)
        generated_ids, encoded_messages = encode_spar(model=model_manager.model, context=tokenized_context, message_bits=to_decode,
                                                      min_token_length=MIN_TOKEN_LENGTH, max_token_length=MAX_TOKEN_LENGTH,
                                                      random_seed=random_number, device=model_manager.device)
        encoded_message = "".join(encoded_messages)
        all_generated_ids.extend(generated_ids)
        m = model_manager.tokenizer.decode(generated_ids)
        final_messages.append(m)
        i += len(encoded_message)
        to_decode = to_decode[len(encoded_message):]
    print(all_generated_ids)
    return final_messages


def encode_step(probs, n_m, k_m):
    r = random.random()
    cumulative_probs = probs.cumsum(0)
    r_i_m = func_mrn(k_m, n_m, r)
    token_index = (cumulative_probs > r_i_m).nonzero()[0].item()

    SE = get_lower_upper_bound(cumulative_probs, token_index)
    temp0 = ceil((SE[0] - r) * n_m)
    temp1 = ceil((SE[1] - r) * n_m)

    if k_m + r * n_m >= n_m:
        k_m = k_m - n_m - temp0
    else:
        k_m = k_m - temp0
    n_m = temp1 - temp0
    return token_index, n_m, k_m


@torch.no_grad()
def encode_spar(model, context, message_bits, min_token_length, max_token_length, device='cuda', block_size=BLOCK_SIZE,
                top_p=TOP_P, random_seed=42):
    context = torch.tensor(context[-CONTEXT_WINDOW_SIZE:], device=device, dtype=torch.long)

    generated_ids = []
    m_index = 0
    k_m = int(message_bits[:block_size], 2)
    n_m = 2 ** block_size
    token_num_generated = 0
    random.seed(random_seed)
    encoded_message = []
    past = None
    prev = context
    message_blocks = len(message_bits) // block_size

    while True:
        probs, indices, past = get_probs_past(model=model,
                                              prev=prev,
                                              past=past,
                                              device=device,
                                              top_p=top_p)

        probs = probs.to(torch.float64)
        token_index, n_m, k_m = encode_step(probs=probs, n_m=n_m, k_m=k_m)
        tokenID = indices[token_index]
        token_num_generated += 1
        if token_num_generated < min_token_length and m_index+block_size < len(message_bits):
            if n_m == 1:
                encoded_message.append(message_bits[m_index:m_index + block_size])
                m_index += block_size
                n_m = 2 ** block_size
                k_m = int(message_bits[m_index:m_index + block_size], 2)
        else:
            if n_m == 1:
                encoded_message.append(message_bits[m_index:m_index + block_size])
                m_index += block_size
                generated_ids.append(tokenID.item())
                break
            if token_num_generated > max_token_length:
                print(
                    f"We have generated more than 12000 tokens,but this block message still not embedded over. This context seems have problem. let's skip it.")
                raise Exception("This context seems have problem.let's skip it.")
        generated_ids.append(tokenID.item())
        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)
    print("Successfully encoded message block!")
    print("Generated ids:", generated_ids)
    print("Encoded message:", encoded_message)
    print("random seed:", random_seed)
    return generated_ids, encoded_message

def process_message_with_checkpoints(
        message_text: str,
        block_size: int = BLOCK_SIZE,
        blocks_per_interval: int = BLOCKS_PER_INTERVAL
) -> str:
    """
    Convert message to binary string with checkpoint markers for verification during decoding.

    Every 'blocks_per_interval' blocks, a checkpoint is inserted. Checkpoint blocks contain
    (block_size - 8) message bits followed by 8 zero bits as a marker. This allows the
    BackCheck decoding algorithm to verify tokenization matches at regular intervals.

    Args:
        message_text: Message string to convert to binary
        block_size: Size of each block in bits (default: BLOCK_SIZE)
        blocks_per_interval: Number of blocks before inserting a checkpoint (default: BLOCKS_PER_INTERVAL)

    Returns:
        Binary string containing message bits with checkpoint markers inserted
    """
    message_bits = string_to_utf8_binary(message_text)
    blocks = []
    idx = 0
    total_msg_len = len(message_bits)
    interval_counter = 0

    while idx < total_msg_len:
        interval_counter += 1

        # Does this block get a checkpoint byte?
        if interval_counter == blocks_per_interval:
            # Only block_size-8 bits from message, then 8 zeroes
            take_len = block_size - 8
            block_bits = message_bits[idx:idx + take_len]
            if len(block_bits) < take_len:
                # Not enough bits, pad up to take_len
                block_bits = block_bits.ljust(take_len, '0')
            block_bits += "00000000"
            blocks.append(block_bits)
            idx += take_len
            interval_counter = 0  # Reset for next interval
        else:
            # Take a full block_size of message bits (or pad final block)
            block_bits = message_bits[idx:idx + block_size]
            if len(block_bits) < block_size:
                block_bits = block_bits.ljust(block_size, '0')
            blocks.append(block_bits)
            idx += block_size

    # If last block was not at exact interval boundary, check if padding needed
    # Not strictly required due to 'ljust' use, but can be explicit

    # Join blocks together
    return ''.join(blocks)
