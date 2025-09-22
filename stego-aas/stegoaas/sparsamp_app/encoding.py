from __future__ import annotations  # no effect in 3.8 but safe

import random
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import torch

from .sparsamp_utils import func_mrn, get_lower_upper_bound, get_probs_past, MODEL, TOKENIZER, DEVICE, \
    string_to_utf8_binary

Edge = Tuple[int, int]  # (next_char_index, token_id)
TokenGraph = Dict[int, List[Edge]]  # char_index -> list of edges


def full_encode(context, message_text, random_seed):
    message_bits = process_message_with_checkpoints(message_text, 32)
    final_messages = []
    i = 0
    tokenized_context = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
    rng = np.random.default_rng(random_seed)
    to_decode = message_bits
    while i < len(message_bits):
        random_number = rng.integers(low=10 ** 15, high=10 ** 16)
        min_tokens = min(100, len(to_decode) // 7)
        generated_ids, encoded_messages = encode_spar(model=MODEL, context=tokenized_context, message_bits=to_decode,
                                                      min_token_length=min_tokens, max_token_length=1000,
                                                      random_seed=random_number, device=DEVICE)
        encoded_message = "".join(encoded_messages)
        m = TOKENIZER.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_messages.append(m)
        i += len(encoded_message)
        to_decode = to_decode[len(encoded_message):]
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
def encode_spar(model, context, message_bits, min_token_length, max_token_length, device='cuda', block_size=32,
                top_p=1.0, random_seed=42):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    generated_ids = []
    m_index = 0
    k_m = int(message_bits[:block_size], 2)
    n_m = 2 ** block_size
    token_num_generated = 0
    random.seed(random_seed)
    encoded_message = []
    past = None
    prev = context

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
        if token_num_generated < min_token_length:
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

    return generated_ids, encoded_message



def process_message_with_checkpoints(
        message_text: str,
        block_size: int = 32,
        blocks_per_interval: int = 4
) -> str:
    """
    Convert message to binary string split into blocks;
    Every 'blocks_per_interval' blocks, the block stores only (block_size - 8) message bits,
    reserving the last 8 bits for a checkpoint (all zeros).
    The final block is padded to full block_size if necessary.
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
