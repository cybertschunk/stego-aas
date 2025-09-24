
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

from dataclasses import dataclass, field

# Type definitions
Edge = Tuple[int, int] # (next_char_index, token_id)
TokenGraph = Dict[int, List[Edge]] # char_index -> list of edges

# ============================================================================
# BACKCHECK IMPLEMENTATION - NEW ADDITION
# ============================================================================

@dataclass
class BackCheckNode:
    """Node in the BackCheck tree for token ambiguity resolution"""

    # Node attributes as specified in requirements
    known_correct: bool = False
    known_wrong: bool = False
    token: int = -1  # Token index
    text: str = ""  # Text representation of the token
    children: List['BackCheckNode'] = field(default_factory=list)
    parent: Optional['BackCheckNode'] = None
    greedy_correct: bool = False  # Part of default tokenization
    likeliness: float = -1.0  # Probability that this node is correct

    # Position tracking
    start_pos: int = 0  # Start position in text
    end_pos: int = 0   # End position in text

    def add_child(self, child: 'BackCheckNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

@dataclass
class BackCheckVerificationResult:
    """Result of BackCheck verification function"""
    decoded_message: str = ""
    correct_tokens: List[int] = field(default_factory=list)
    visited_tokens: List[int] = field(default_factory=list)
    success: bool = False
    random_state: any = None

class BackCheckTree:
    """BackCheck tree for token ambiguity resolution"""

    def __init__(self, stego_text: str, tokenizer, model):
        self.stego_text = stego_text
        self.tokenizer = tokenizer
        self.model = model
        self.root = BackCheckNode()  # Empty root node
        self.all_nodes = []  # Keep track of all nodes

        # Build the initial tree with default tokenization
        self._build_initial_tree()

    def _build_initial_tree(self):
        """Build initial tree with default tokenization as greedy path"""
        # Get default (greedy) tokenization
        default_tokens = self.tokenizer.encode(self.stego_text, add_special_tokens=False)

        # Create nodes for default tokenization path
        current_pos = 0
        current_node = self.root

        for i, token_id in enumerate(default_tokens):
            token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            end_pos = current_pos + len(token_text)

            # Create node for this token
            node = BackCheckNode(
                token=token_id,
                text=token_text,
                start_pos=current_pos,
                end_pos=end_pos,
                greedy_correct=True,  # This is part of default tokenization
                likeliness=0.5  # Default likelihood
            )

            current_node.add_child(node)
            self.all_nodes.append(node)
            current_node = node
            current_pos = end_pos

    def explore_node(self, node: BackCheckNode):
        """
        Explore function: determine all possible tokens that could follow after this node
        and calculate their probabilities using the model
        """
        if node.end_pos >= len(self.stego_text):
            return  # Already at end of text

        # Get remaining text to match
        remaining_text = self.stego_text[node.end_pos:]

        # Find all possible tokens that could continue from this position
        possible_continuations = []

        # Use efficient approach to find valid continuations
        sorted_tokens = self._prepare_tokens_for_prefix(remaining_text)

        # Check each token to see if it could be a valid continuation
        for token_str, token_id in sorted_tokens:
            if remaining_text.startswith(token_str):
                possible_continuations.append((token_id, token_str, len(token_str)))

        if not possible_continuations:
            return

        # Get probability distribution for these tokens using the model
        token_probs = self._get_token_probabilities_from_model(
            self.stego_text[:node.end_pos],
            [tid for tid, _, _ in possible_continuations]
        )

        # Create child nodes for each possible continuation
        for token_id, token_text, token_len in possible_continuations:
            child_node = BackCheckNode(
                token=token_id,
                text=token_text,
                start_pos=node.end_pos,
                end_pos=node.end_pos + token_len,
                greedy_correct=False,  # These are alternatives to greedy
                likeliness=token_probs.get(token_id, 0.0001)  # Small default probability
            )

            node.add_child(child_node)
            self.all_nodes.append(child_node)

    def _prepare_tokens_for_prefix(self, remaining_text: str) -> List[Tuple[str, int]]:
        """Prepare sorted list of tokens that could match the remaining text prefix"""
        # Use existing _prepare_tokens function but optimized for prefix matching
        allowed = set(remaining_text)
        tups = []

        # Limit vocabulary size for efficiency - focus on most common tokens
        vocab_limit = min(self.tokenizer.vocab_size, 10000)

        for tid in range(vocab_limit):
            try:
                s = self.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                if s and len(s) <= len(remaining_text):
                    if all(ch in allowed or ch.isspace() for ch in s):
                        tups.append((s, tid))
            except:
                continue

        return sorted(tups)

    def _get_token_probabilities_from_model(self, context: str, token_ids: List[int]) -> Dict[int, float]:
        """Get probabilities for specific tokens given context using model"""
        try:
            # Convert context to tensor
            context_tokens = self.tokenizer.encode(context, return_tensors='pt')
            if hasattr(context_tokens, 'to'):
                context_tokens = context_tokens.to(DEVICE)

            # Use existing get_probs_past function
            probs, indices, past = get_probs_past(
                model=self.model,
                prev=context_tokens,
                past=None,
                device=DEVICE,
                top_p=1.0
            )

            # Extract probabilities for specific token_ids
            result_probs = {}
            for token_id in token_ids:
                token_idx = torch.where(indices == token_id)[0]
                if len(token_idx) > 0:
                    result_probs[token_id] = float(probs[token_idx[0]].item())
                else:
                    result_probs[token_id] = 1e-10  # Very small probability

            return result_probs

        except Exception as e:
            print(f"Error getting token probabilities: {e}")
            # Fallback to uniform probabilities
            prob_per_token = 1.0 / len(token_ids) if token_ids else 0.0
            return {token_id: prob_per_token for token_id in token_ids}

class BackCheckDecoder:
    """BackCheck decoder that integrates with existing decoding functions"""

    def __init__(self, model, tokenizer, context, device=DEVICE, **decode_params):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = device
        self.decode_params = decode_params

    def verify_path(self, token_path: List[int]) -> BackCheckVerificationResult:
        """Verification function using existing try_blind_decoding"""
        try:
            # Use existing try_blind_decoding function
            success, final_state, last_correct_idx = try_blind_decoding(
                token_sequence=token_path,
                last_verified_state=None,
                model=self.model,
                context=self.context,
                device=self.device,
                block_size=self.decode_params.get('block_size', 32),
                top_p=self.decode_params.get('top_p', 1.0),
                random_seed=self.decode_params.get('random_seed', 42),
                blocks_per_interval=self.decode_params.get('blocks_per_interval', 4),
                get_probs_past_func=self.decode_params.get('get_probs_past_func', get_probs_past)
            )

            if success and final_state:
                decoded_message = final_state.get_discovered_message_as_string()
                correct_tokens = token_path[:last_correct_idx + 1] if last_correct_idx >= 0 else []

                return BackCheckVerificationResult(
                    decoded_message=decoded_message,
                    correct_tokens=correct_tokens,
                    visited_tokens=token_path.copy(),
                    success=True,
                    random_state=getattr(final_state, 'random_state_data', None)
                )
            else:
                return BackCheckVerificationResult(
                    decoded_message="",
                    correct_tokens=token_path[:last_correct_idx + 1] if last_correct_idx >= 0 else [],
                    visited_tokens=token_path.copy(),
                    success=False,
                    random_state=None
                )

        except Exception as e:
            print(f"Error in BackCheck path verification: {e}")
            return BackCheckVerificationResult(
                decoded_message="",
                correct_tokens=[],
                visited_tokens=token_path.copy(),
                success=False,
                random_state=None
            )

    def backcheck_decode_tree(self, tree: BackCheckTree, max_attempts: int = 100) -> Optional[Tuple[List[int], str]]:
        """Main BackCheck algorithm implementation"""
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Find path through tree following priority rules
            path_tokens = self._find_path_through_tree(tree)

            if not path_tokens:
                print(f"BackCheck: No valid path found after {attempts} attempts")
                break

            # Verify the path using existing decoding
            result = self.verify_path(path_tokens)

            # Update tree based on verification results
            self._update_tree_from_verification(tree, result, path_tokens)

            if result.success:
                print(f"✅ BackCheck succeeded after {attempts} attempts!")
                return path_tokens, result.decoded_message

            if attempts % 10 == 0:
                print(f"BackCheck: Attempt {attempts}, continuing search...")

        return None

    def _find_path_through_tree(self, tree: BackCheckTree) -> Optional[List[int]]:
        """Find path through tree following priority rules"""

        def search_recursive(current_node: BackCheckNode) -> Optional[List[int]]:
            # If we've reached end of text, return empty path (success)
            if current_node.end_pos >= len(tree.stego_text):
                return []

            # If node has no children, explore to create them
            if len(current_node.children) == 0 and current_node != tree.root:
                tree.explore_node(current_node)

            # Get available children (not marked as wrong)
            available_children = [child for child in current_node.children if not child.known_wrong]

            if not available_children:
                return None  # No valid children, backtrack

            # Sort children by priority
            sorted_children = self._sort_children_by_priority(available_children)

            # Try each child in priority order
            for child in sorted_children:
                child_path = search_recursive(child)
                if child_path is not None:
                    return [child.token] + child_path

            return None  # No child succeeded

        return search_recursive(tree.root)

    def _sort_children_by_priority(self, children: List[BackCheckNode]) -> List[BackCheckNode]:
        """Sort children by priority: known_correct > greedy_correct > likeliness"""

        def priority_key(node: BackCheckNode):
            if node.known_correct:
                return (0, 0)  # Highest priority
            elif node.greedy_correct and not node.known_wrong:
                return (1, 0)  # Second priority
            elif not node.known_wrong:
                return (2, -node.likeliness)  # Third priority (negative for desc sort)
            else:
                return (3, 0)  # Lowest priority

        return sorted(children, key=priority_key)

    def _update_tree_from_verification(self, tree: BackCheckTree, result: BackCheckVerificationResult, path_tokens: List[int]):
        """Update tree nodes based on verification results"""

        # Mark correct tokens as known_correct
        for correct_token in result.correct_tokens:
            for node in tree.all_nodes:
                if node.token == correct_token and not node.known_correct:
                    node.known_correct = True
                    break

        # If verification failed, mark last visited token as wrong
        if not result.success and result.visited_tokens:
            last_failed_token = result.visited_tokens[-1]

            for node in tree.all_nodes:
                if (node.token == last_failed_token and
                    not node.known_correct and
                    not node.known_wrong):
                    node.known_wrong = True
                    break

def backcheck_decode_single_message(message: str, context: str, random_seed: int) -> str:
    """
    Decode a single message using BackCheck algorithm
    """
    try:
        # First try BackCheck approach
        print(f"🌳 Trying BackCheck decoding for message length {len(message)}")

        # Build BackCheck tree
        tree = BackCheckTree(message, TOKENIZER, MODEL)

        # Set up decoder parameters
        decoder_params = {
            'block_size': 32,
            'top_p': 1.0,
            'random_seed': random_seed,
            'blocks_per_interval': 4,
            'get_probs_past_func': get_probs_past
        }

        # Create decoder with context
        context_tensor = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
        decoder = BackCheckDecoder(MODEL, TOKENIZER, context_tensor, DEVICE, **decoder_params)

        # Try BackCheck decoding
        result = decoder.backcheck_decode_tree(tree, max_attempts=50)

        if result:
            tokenization, decoded_message = result
            print(f"✅ BackCheck successful: '{decoded_message}'")
            return decoded_message.rstrip("\x00")
        else:
            print("⚠️ BackCheck failed, falling back to linear approach")

            # Fallback to existing linear approach
            tokenization, decoded_message = linear_blind_backcheck(
                stego_text=message,
                model=MODEL,
                tokenizer=TOKENIZER,
                context=context_tensor,
                device=DEVICE,
                random_seed=random_seed,
                max_attempts=1000
            )

            if decoded_message:
                print(f"✅ Linear fallback successful: '{decoded_message}'")
                return decoded_message.rstrip("\x00")
            else:
                print("❌ Both BackCheck and linear approaches failed")
                return ""

    except Exception as e:
        print(f"Error in BackCheck decoding: {e}")

        # Fallback to decode_spar if BackCheck fails
        try:
            tokenized_message = TOKENIZER.encode(message, return_tensors='pt')[0].tolist()
            context_tensor = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
            decoded_message = decode_spar(model=MODEL, device=DEVICE, random_seed=random_seed,
                                        context=context_tensor, generated_ids=tokenized_message)
            utf8_string = utf8_binary_to_string("".join(decoded_message))
            return utf8_string.rstrip("\x00")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return ""

# ============================================================================
# EXISTING CODE - PRESERVED WITH MINIMAL CHANGES
# ============================================================================

class BlindDecodingState:
    """State for blind decoding"""
    def __init__(self):
        self.random_state_data = None
        self.discovered_message_bits = ""
        self.decoded_blocks = []
        self.completed_blocks = 0
        self.prev = None
        self.past = None

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
            return f""

def _prepare_tokens(tokenizer, text):
    """
    Prepare sorted list of (decoded_string, token_id) tuples.
    Only includes tokens whose characters appear in the input text.
    """
    allowed = set(text) # every Unicode character in the text
    tups = [] # (decoded_string, token_id)
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and all(ch in allowed or ch.isspace() for ch in s):
            tups.append((s, tid))
    return sorted(tups) # lexicographic order

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
        edges_sorted = sorted(edges, key=lambda x: x[1]) # Sort by token_id (lower = more common)

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
    n_m = 2 ** block_size
    k_m = 0
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []
    current_interval_counter = 0
    token_count = 0
    start_token_idx = 0
    if last_verified_state is None:
        state = BlindDecodingState()
        state.past = None
        state.prev = context
        random.seed(random_seed)
        state.save_random_state()
        state.token_count = 0
        state.prev = context
    else:
        state = copy.deepcopy(last_verified_state)
        state.restore_random_state()
        start_token_idx = state.token_count
    try:
        for token_idx in range(start_token_idx, len(token_sequence)):
            tokenID = token_sequence[token_idx]
            r = random.random()

            probs, indices, state.past = get_probs_past(
                model=model, prev=state.prev, past=state.past, device=device, top_p=top_p
            )

            probs = probs.to(torch.float64)
            cumulative_probs = probs.cumsum(0)

            token_index = torch.where(indices == tokenID)[0]
            SE = get_lower_upper_bound(cumulative_probs,token_index)
            temp0 = ceil((SE[0] - r) * n_m)
            temp1 = ceil((SE[1] - r) * n_m)

            n_m = temp1 - temp0
            temp0_arr.append(temp0)
            temp1_arr.append(temp1)
            n_m_arr.append(n_m)

            if n_m <= 0:
                return False, state, token_idx - 1

            token_count += 1

            if n_m == 1:
                current_interval_counter += 1
                is_checkpoint = (current_interval_counter == blocks_per_interval)

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
                        return False, state, token_idx - 1
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=True)
                    state.current_interval_counter = 0
                else:
                    state.add_discovered_block(decoded_block_bits, is_checkpoint=False)

                temp0_arr = []
                temp1_arr = []
                n_m_arr = []
                n_m = 2 ** block_size
                k_m = 0
                state.save_random_state()

            state.prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)

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
                           context: str, # Accept string context
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
    """
    Main entry point for decoding - now with BackCheck integration

    Preserves the original function signature while adding BackCheck functionality
    """
    final_messages = []
    context_tokens = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
    rng = np.random.default_rng(random_seed)

    for i, message in enumerate(messages):
        print(f"\n🔍 Decoding message {i+1}/{len(messages)}")
        random_number = rng.integers(low=10**15, high=10**16)

        # Try BackCheck decoding first
        decoded_message = backcheck_decode_single_message(message, context, random_number)

        if decoded_message:
            final_messages.append(decoded_message)
        else:
            print(f"⚠️ All decoding approaches failed for message {i+1}")
            final_messages.append("")

    final_message = "".join(final_messages)
    return final_message
