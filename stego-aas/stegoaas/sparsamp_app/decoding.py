
from __future__ import annotations

import copy
import random
from bisect import bisect_left, bisect_right
from math import ceil
from typing import List, Dict, Optional, Tuple, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .sparsamp_utils import TOKENIZER, MODEL, DEVICE, get_probs_past, dec2bin, get_lower_upper_bound, \
    utf8_binary_to_string

# Type definitions
Edge = Tuple[int, int]  # (next_char_index, token_id)
TokenGraph = Dict[int, List[Edge]]  # char_index -> list of edges


class BlindDecodingState:
    """State for blind decoding"""

    def __init__(self):
        self.random_state_data = None
        self.n_m = None
        self.k_m = None
        self.past = None
        self.block_size = None
        self.token_count = 0
        self.blocks_per_interval = 4
        self.current_interval_counter = 0
        self.discovered_message_bits = ""
        self.decoded_blocks = []
        self.completed_blocks = 0
        self.temp0_arr = []
        self.temp1_arr = []
        self.n_m_arr = []

    def save_random_state(self):
        self.random_state_data = random.getstate()

    def restore_random_state(self):
        if self.random_state_data is not None:
            random.setstate(self.random_state_data)

    def add_discovered_block(self, block_bits: str, is_checkpoint: bool):
        self.decoded_blocks.append(block_bits)
        self.completed_blocks += 1

        if is_checkpoint:
            message_content = block_bits[:self.block_size - 8]
            self.discovered_message_bits += message_content
        else:
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
            return f"<DECODE_ERROR: {self.discovered_message_bits[:100]}...>"


def _prepare_tokens(tokenizer, text):
    """
    Prepare sorted list of (decoded_string, token_id) tuples.
    Only includes tokens whose characters appear in the input text.
    """
    allowed = set(text)  # every Unicode character in the text
    tups = []           # (decoded_string, token_id)

    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and all(ch in allowed or ch.isspace() for ch in s):
            tups.append((s, tid))

    return sorted(tups)  # lexicographic order


def _candidates(sorted_tokens, prefix):
    """
    Return the slice of tokens whose decoded string starts with `prefix`.
    Uses binary search for O(log n) lookup.
    """
    lo = bisect_left(sorted_tokens, (prefix, -1))
    hi = bisect_right(sorted_tokens, (prefix + '\uffff', float('inf')))
    return sorted_tokens[lo:hi]


def build_token_graph(text, tokenizer):
    """
    Build graph of all possible tokenizations using your exact approach.
    This runs in O(n) time where n is the length of the text.

    Returns:
        TokenGraph: dict mapping position -> [(next_position, token_id), ...]
    """
    print(f"Building linear token graph for text of length {len(text)}...")

    # Your optimization: prepare sorted tokens
    tokens_sorted = _prepare_tokens(tokenizer, text)
    print(f"Found {len(tokens_sorted)} valid tokens")

    n = len(text)
    graph = {i: [] for i in range(n + 1)}
    reachable = [False] * (n + 1)
    reachable[0] = True

    # Your linear algorithm: O(n) outer loop, O(12) inner loop = O(n) total
    for i in range(n):
        if not reachable[i]:
            continue

        # Your optimization: longest prefix we ever need to query = longest token (≈12 for GPT-2)
        for L in range(1, min(12, n - i) + 1):
            prefix = text[i:i+L]

            # Your binary search optimization
            candidates = _candidates(tokens_sorted, prefix)

            for tok_str, tid in candidates:
                if text.startswith(tok_str, i):
                    j = i + len(tok_str)
                    graph[i].append((j, tid))
                    reachable[j] = True

            # Your early stopping optimization
            if not candidates:
                break

    total_edges = sum(len(edges) for edges in graph.values())
    print(f"Graph built with {total_edges} edges in O(n) time")
    return graph


def prioritized_path_generator(graph: TokenGraph, end_pos: int, tokenizer, text: str) -> Iterable[List[int]]:
    """
    Generate complete tokenization paths in priority order using your graph.
    This directly generates paths without building a tree.
    """

    # First, yield the greedy tokenization
    greedy_tokens = tokenizer.encode(text, add_special_tokens=False)

    # Verify greedy path exists in graph and yield it first
    pos = 0
    greedy_path = []
    greedy_valid = True

    for token_id in greedy_tokens:
        found = False
        for next_pos, tid in graph.get(pos, []):
            if tid == token_id:
                greedy_path.append(token_id)
                pos = next_pos
                found = True
                break
        if not found:
            greedy_valid = False
            break

    if greedy_valid and pos == end_pos:
        print(f"🎯 Yielding greedy tokenization first: {len(greedy_path)} tokens")
        yield greedy_path

    # Then use DFS to generate all other complete paths
    def dfs_generate_paths(current_pos: int, current_path: List[int], visited_paths: set):
        if current_pos == end_pos:
            path_tuple = tuple(current_path)
            if path_tuple not in visited_paths:
                visited_paths.add(path_tuple)
                yield current_path.copy()
            return

        # Get all possible next tokens from current position
        edges = graph.get(current_pos, [])

        # Sort edges by token likelihood (simple heuristic)
        edges_sorted = sorted(edges, key=lambda x: x[1])  # Sort by token_id (lower = more common)

        for next_pos, token_id in edges_sorted:
            current_path.append(token_id)
            yield from dfs_generate_paths(next_pos, current_path, visited_paths)
            current_path.pop()

    # Generate remaining paths
    visited_paths = {tuple(greedy_path)} if greedy_valid else set()
    yield from dfs_generate_paths(0, [], visited_paths)

@torch.no_grad()
def try_blind_decoding(token_sequence: List[int],
                      last_verified_state: Optional[BlindDecodingState],
                      model: PreTrainedModel,
                      context: torch.Tensor,
                      device: str = 'cuda',
                      block_size: int = 32,
                      top_p: float = 1.0,
                      random_seed: int = 42,
                      blocks_per_interval: int = 4,
                      get_probs_past_func=None) -> Tuple[bool, Optional[BlindDecodingState], int]:
    """Try to decode a token sequence without knowing the original message"""

    if last_verified_state is None:
        state = BlindDecodingState()
        state.n_m = 2 ** block_size
        state.k_m = 0
        state.past = None
        state.block_size = block_size
        state.blocks_per_interval = blocks_per_interval
        state.current_interval_counter = 0
        state.token_count = 0

        random.seed(random_seed)
        state.save_random_state()

        prev = context
        start_token_idx = 0
    else:
        state = copy.deepcopy(last_verified_state)
        state.restore_random_state()
        start_token_idx = state.token_count

        if len(token_sequence) > start_token_idx and start_token_idx > 0:
            last_token = token_sequence[start_token_idx - 1]
            prev = torch.tensor([last_token], device=device, dtype=torch.long).unsqueeze(0)
        else:
            prev = context

    try:
        for token_idx in range(start_token_idx, len(token_sequence)):
            tokenID = token_sequence[token_idx]
            r = random.random()


            probs, indices, state.past = get_probs_past(
                model=model, prev=prev, past=state.past, device=device, top_p=top_p
            )

            probs = probs.to(torch.float64)
            cumulative_probs = probs.cumsum(0)

            token_index = torch.where(indices == tokenID)[0]
            SE = get_lower_upper_bound(cumulative_probs,token_index)

            temp0 = ceil((SE[0] - r) * state.n_m)
            temp1 = ceil((SE[1] - r) * state.n_m)
            state.temp0_arr.append(temp0)

            state.n_m = temp1 - temp0
            state.n_m_arr.append(state.n_m)

            if state.n_m <= 0:
                return False, state, token_idx - 1
            state.temp1_arr.append(temp1)
            state.temp0_arr.append(temp0)
            state.n_m_arr.append(state.n_m)

            state.token_count += 1

            if state.n_m == 1:
                state.current_interval_counter += 1
                is_checkpoint = (state.current_interval_counter == state.blocks_per_interval)

                count = len(state.temp0_arr) - 2
                state.k_m = state.temp0_arr[count+1]
                while count >= 0:
                    n_m_new = state.n_m_arr[count]
                    state.k_m = state.temp0_arr[count] + ((state.k_m + n_m_new) % n_m_new)
                    count -= 1
                state.k_m = (state.k_m + 2 ** block_size) % (2 ** block_size)
                decoded_block_bits = dec2bin(state.k_m, block_size)

                if is_checkpoint:
                    if not decoded_block_bits.endswith('00000000'):
                        return False, state, token_idx - 1
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=True)
                    state.current_interval_counter = 0
                else:
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=False)

                state.temp0_arr = []
                state.temp1_arr = []
                state.n_m_arr = []
                state.n_m = 2 ** block_size
                state.k_m = 0

                state.save_random_state()
            prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

        return True, state, len(token_sequence) - 1

    except Exception as e:
        print(f"Exception during decoding: {e}")
        return False, state, start_token_idx - 1


def linear_blind_backcheck(stego_text: str,
                          model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizer,
                          context: torch.Tensor,
                          device: str = 'cuda',
                          block_size: int = 32,
                          top_p: float = 1.0,
                          random_seed: int = 42,
                          blocks_per_interval: int = 4,
                          get_probs_past_func=None,
                          max_attempts: int = 10000) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    ULTRA-FAST blind backcheck using your linear O(n) graph approach.

    This is the fastest possible approach:
    - O(n) graph construction where n = text length
    - Direct path generation without tree overhead
    - Immediate testing of each tokenization
    """

    print(f"🚀 LINEAR O(n) blind backcheck for text of length {len(stego_text)}")

    import time
    start_time = time.time()

    # Build the linear token graph - O(n) complexity!
    graph = build_token_graph(stego_text, tokenizer)
    graph_time = time.time() - start_time
    print(f"✅ Linear graph built in {graph_time:.3f} seconds")

    # Generate and test tokenizations directly from the graph
    print("🔍 Testing tokenizations directly from graph...")

    attempts = 0
    for tokenization in prioritized_path_generator(graph, len(stego_text), tokenizer, stego_text):
        attempts += 1

        if attempts > max_attempts:
            print(f"⏰ Reached maximum attempts ({max_attempts}), stopping")
            break

        if attempts % 100 == 0:
            elapsed = time.time() - start_time
            print(f"📊 Tested {attempts} tokenizations in {elapsed:.1f}s...")

        # Test this tokenization immediately
        success, final_state, last_correct = try_blind_decoding(
            tokenization, None, model, context,
            device, block_size, top_p, random_seed, blocks_per_interval,
            get_probs_past_func
        )

        if success and final_state:
            total_time = time.time() - start_time
            discovered_message = final_state.get_discovered_message_as_string()

            print(f"\n🎉 SUCCESS after {attempts} attempts in {total_time:.2f}s!")
            print(f"⚡ Graph construction: {graph_time:.3f}s")
            print(f"⚡ Search time: {total_time - graph_time:.2f}s")
            print(f"📝 Discovered message: '{discovered_message}'")

            return tokenization, discovered_message

    total_time = time.time() - start_time
    print(f"\n❌ FAILED after testing {attempts} tokenizations in {total_time:.2f}s")
    return None, None


# Convenience wrapper
def ultra_linear_backcheck(stego_text: str,
                          model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizer,
                          context: str,  # Accept string context
                          device: str = 'cuda',
                          random_seed: int = 42,
                          get_probs_past_func=None,
                          **kwargs) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Ultra-convenient wrapper using your linear O(n) approach.
    """

    # Convert context string to tensor
    context_tensor = tokenizer.encode(context, return_tensors='pt').to(device)

    return linear_blind_backcheck(
        stego_text=stego_text,
        model=model,
        tokenizer=tokenizer,
        context=context_tensor,
        device=device,
        random_seed=random_seed,
        get_probs_past_func=get_probs_past_func,
        **kwargs
    )

@torch.no_grad()
def decode_spar(model, generated_ids, context, device='cuda', block_size=32, top_p=1.0, random_seed=42):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    random.seed(random_seed)
    message = []
    n_m = 2 ** block_size
    k_m = 0
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []
    past = None
    prev = context

    for tokenID in generated_ids:
        r = random.random()
        probs, indices, past = get_probs_past(model=model,
                                              prev=prev,
                                              past=past,
                                              device=device,
                                              top_p=top_p)
        probs = probs.to(torch.float64)
        cumulative_probs = probs.cumsum(0)

        token_index = torch.where(indices == tokenID)[0]
        SE = get_lower_upper_bound(cumulative_probs, token_index)

        temp0 = ceil((SE[0] - r) * n_m)
        temp1 = ceil((SE[1] - r) * n_m)

        n_m = temp1 - temp0
        temp0_arr.append(temp0)
        temp1_arr.append(temp1)
        n_m_arr.append(n_m)

        if n_m == 1:
            count = len(temp0_arr) - 2
            k_m = temp0_arr[count + 1]
            while count >= 0:
                n_m_new = n_m_arr[count]
                k_m = temp0_arr[count] + ((k_m + n_m_new) % n_m_new)
                count -= 1
            k_m = (k_m + 2 ** block_size) % 2 ** block_size
            temp0_arr = []
            temp1_arr = []
            n_m_arr = []
            message.append(dec2bin(k_m, block_size))
            n_m = 2 ** block_size
        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

    return message



def full_decode(context, messages, random_seed):
    final_messages = []
    context = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
    rng = np.random.default_rng(random_seed)
    for message in messages:
        random_number = rng.integers(low=10**15, high=10**16)
        tokenized_message = TOKENIZER.encode(message, return_tensors='pt')[0].tolist()
        decoded_message = decode_spar(model=MODEL,device=DEVICE,random_seed=random_number,context=context,generated_ids=tokenized_message)
        utf8_string = utf8_binary_to_string("".join(decoded_message))
        uft8_string_stripped = utf8_string.rstrip("\x00")
        final_messages.append(uft8_string_stripped)
    final_message = "".join(final_messages)
    return final_message




