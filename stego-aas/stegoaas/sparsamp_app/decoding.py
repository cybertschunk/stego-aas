
from __future__ import annotations

import copy
import logging
import random
from math import ceil
from typing import List, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel

from .constants import (
    BLOCK_SIZE, BLOCKS_PER_INTERVAL, TOP_P,
    RANDOM_SEED_MIN, RANDOM_SEED_MAX
)
from .backcheck import backcheck_decode_single_message
from .sparsamp_utils import get_probs_past, dec2bin, get_lower_upper_bound

logger = logging.getLogger(__name__)

class BlindDecodingState:
    """State for blind decoding"""
    def __init__(self):
        self.random_state_data = None
        self.discovered_message_bits = ""
        self.decoded_blocks = []
        self.backcheck_count = -1
        self.prev = None
        self.past = None
        self.token_count = 0
        self.current_tokens = []

    def save_random_state(self) -> None:
        self.random_state_data = random.getstate()

    def restore_random_state(self) -> None:
        if self.random_state_data is not None:
            random.setstate(self.random_state_data)

    def add_discovered_block(self, block_bits: str, is_checkpoint: bool) -> None:
        self.decoded_blocks.append(block_bits)
        self.discovered_message_bits += block_bits

    def get_discovered_message_as_string(self) -> str:
        try:
            if len(self.discovered_message_bits) % 8 != 0:
                padded_bits = self.discovered_message_bits.ljust(
                    ((len(self.discovered_message_bits) + 7) // 8) * 8, '0'
                )
            else:
                padded_bits = self.discovered_message_bits

            message_bytes = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte_bits = padded_bits[i:i+8]
                if len(byte_bits) == 8:
                    byte_value = int(byte_bits, 2)
                    if byte_value != 0:
                        message_bytes.append(byte_value)

            decoded_string = message_bytes.decode('utf-8', errors='replace')
            return decoded_string.rstrip('\x00')
        except Exception as e:
            return f""


@torch.no_grad()
def try_blind_decoding(token_sequence: List[int],
                       last_verified_state: BlindDecodingState,
                       model: PreTrainedModel,
                       device: str = 'cuda',
                       block_size: int = BLOCK_SIZE,
                       top_p: float = TOP_P,
                       blocks_per_interval: int = BLOCKS_PER_INTERVAL) -> Tuple[bool, BlindDecodingState, int]:
    """Try to decode a token sequence without knowing the original message"""
    n_m = 2 ** block_size
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []

    state = copy.deepcopy(last_verified_state)
    state.restore_random_state()
    start_token_idx = len(state.current_tokens)
    try:
        for token_idx in range(start_token_idx, len(token_sequence)):
            tokenID = token_sequence[token_idx]
            r = random.random()

            probs, indices, state.past = get_probs_past(
                model=model, prev=state.prev, past=state.past, device=device, top_p=top_p
            )

            state.prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)
            state.token_count += 1
            state.current_tokens.append(tokenID)

            probs = probs.to(torch.float64)
            cumulative_probs = probs.cumsum(0)
            matches = (indices == tokenID).nonzero(as_tuple=True)[0]
            if matches.numel() == 0:
                # Kein Token-Index gefunden — Abbruch mit fehlerhaftem Ergebnis
                return False, last_verified_state, state.token_count

            # Verwende den ersten gefundenen Index als Integer
            token_index = int(matches[0].item())
            SE = get_lower_upper_bound(cumulative_probs,token_index)
            temp0 = ceil((SE[0] - r) * n_m)
            temp1 = ceil((SE[1] - r) * n_m)

            n_m = temp1 - temp0
            temp0_arr.append(temp0)
            temp1_arr.append(temp1)
            n_m_arr.append(n_m)



            if n_m <= 0:
                return False, last_verified_state, state.token_count


            if n_m == 1:
                state.backcheck_count += 1
                is_checkpoint = (state.backcheck_count == blocks_per_interval)

                count = len(temp0_arr) - 2
                k_m = temp0_arr[count+1]

                while count >= 0:
                    n_m_new = n_m_arr[count]
                    k_m = temp0_arr[count] + ((k_m + n_m_new) % n_m_new)
                    count -= 1

                k_m = (k_m + 2 ** block_size) % (2 ** block_size)
                decoded_block_bits = dec2bin(k_m, block_size)

                if is_checkpoint:
                    if not decoded_block_bits.endswith('00000000'):
                        return False, last_verified_state, state.token_count
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=True)
                    state.backcheck_count = 0
                    state.save_random_state()
                    last_verified_state = copy.deepcopy(state)
                else:
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=False)

                temp0_arr = []
                temp1_arr = []
                n_m_arr = []
                n_m = 2 ** block_size
                k_m = 0




        return True, state, len(token_sequence) - 1

    except Exception as e:
        logger.error(f"Exception during decoding: {e}")
        #raise e
        return False, last_verified_state, state.token_count


def init_decoding_state(context: torch.Tensor, initial_backcheck_count: int, random_seed: int) -> BlindDecodingState:
    """Initialize a new decoding state with given parameters"""
    state = BlindDecodingState()
    state.past = None
    state.prev = context
    random.seed(int(random_seed))  # Convert to Python int for Python 3.13 compatibility
    state.save_random_state()
    state.token_count = 0
    state.backcheck_count = initial_backcheck_count
    state.prev = context
    return state


def full_decode(context, messages, random_seed):
    """
    Main entry point for decoding - now with BackCheck integration

    Returns:
        Tuple of (decoded_message, attempts_list) where attempts_list contains
        the number of BackCheck attempts for each message
    """
    final_messages = []
    attempts_list = []
    backcheck_count = 0
    rng = np.random.default_rng(random_seed)

    for i, message in enumerate(messages):
        logger.info(f"Decoding message {i+1}/{len(messages)}")
        random_number = rng.integers(low=RANDOM_SEED_MIN, high=RANDOM_SEED_MAX)

        # Try BackCheck decoding first
        decoded_message, backcheck_count, attempts = backcheck_decode_single_message(message, context, random_number, backcheck_count)

        if decoded_message:
            final_messages.append(decoded_message)
            attempts_list.append(attempts)
        else:
            logger.warning(f"All decoding approaches failed for message {i+1}")
            final_messages.append("")
            attempts_list.append(0)

    final_message = "".join(final_messages)
    return final_message, attempts_list
