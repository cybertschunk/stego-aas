
from __future__ import annotations

import copy
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from math import ceil
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel

from .sparsamp_utils import TOKENIZER, MODEL, DEVICE, get_probs_past, dec2bin, get_lower_upper_bound

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
    is_explored = False
    known_correct: bool = False
    known_wrong: bool = False
    token: int = -1  # Token index
    children: List['BackCheckNode'] = field(default_factory=list)
    parent: Optional['BackCheckNode'] = None
    greedy_correct: bool = False  # Part of default tokenization
    likeliness: float = -1.0  # Probability that this node is correct

    def add_child(self, child: 'BackCheckNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

    def get_path_tokens(self) -> List[int]:
        """Get the token path from root to this node"""
        tokens = []
        current = self
        while current.token != -1: # stop at root node
            tokens.append(current.token)
            current = current.parent
        tokens.reverse()
        return tokens

@dataclass
class BackCheckVerificationResult:
    """Result of BackCheck verification function"""
    decoded_message: str = ""
    correct_tokens: List[int] = field(default_factory=list)
    visited_tokens: List[int] = field(default_factory=list)
    success: bool = False
    last_verified_state: BlindDecodingState = None
    backcheck_count: int = -1

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
        greedy_tokens = self.tokenizer.encode(self.stego_text)

        # Create nodes for default tokenization path
        current_node = self.root

        self._build_greedy_path(current_node, greedy_tokens)

    def _build_greedy_path(self, current_node, greedy_tokens):
        for i, token_id in enumerate(greedy_tokens):

            # Create node for this token
            node = BackCheckNode(
                token=token_id,
                greedy_correct=True,  # This is part of default tokenization
                likeliness=0.5  # Default likelihood
            )

            current_node.add_child(node)
            self.all_nodes.append(node)
            current_node = node

    def explore_node(self, node: BackCheckNode):
        """
        Explore function: determine all possible tokens that could follow after this node
        and calculate their probabilities using the model
        """

        # Get remaining text to match
        current_text = TOKENIZER.decode(node.get_path_tokens()).strip()
        remaining_text = self.stego_text.strip()[len(current_text):]
        if len(node.children) > 0:
            self.explore_model_based(node, remaining_text)
            node.is_explored = True
            return

        greedy_tokens = self.tokenizer.encode(remaining_text, add_special_tokens=False)
        self._build_greedy_path(node,greedy_tokens)


    def explore_model_based(self, node: BackCheckNode, remaining_text: str):

        # Find all possible tokens that could continue from this position
        possible_continuations = []
        # Use efficient approach to find valid continuations
        sorted_tokens = self._prepare_tokens_for_prefix(remaining_text)

        # Check each token to see if it could be a valid continuation
        for token_str, token_id in sorted_tokens:
            test_tokens = node.get_path_tokens() + [token_id]
            test_text = TOKENIZER.decode(test_tokens)
            if self.stego_text.strip().startswith(test_text.strip()) and token_id not in [child.token for child in node.children]:
                possible_continuations.append((token_id, token_str, len(token_str)))

        if not possible_continuations:
            return

        # Get probability distribution for these tokens using the model
        token_probs = self._get_token_probabilities_from_model(
            node.get_path_tokens(),
            [tid for tid, _, _ in possible_continuations]
        )

        # Create child nodes for each possible continuation
        for token_id, token_text, token_len in possible_continuations:
            child_node = BackCheckNode(
                token=token_id,
                greedy_correct=False,  # These are alternatives to greedy
                likeliness=token_probs.get(token_id, 1e-10)  # Small default probability
            )

            node.add_child(child_node)
            self.all_nodes.append(child_node)

    def _prepare_tokens_for_prefix(self, remaining_text: str) -> List[Tuple[str, int]]:
        """Prepare sorted list of tokens that could match the remaining text prefix"""
        # Use existing _prepare_tokens function but optimized for prefix matching
        allowed = set(remaining_text)
        tups = []

        for tid in range(self.tokenizer.vocab_size):
            try:
                s = self.tokenizer.decode([tid]).strip()
                if s and len(s) <= len(remaining_text):
                    if all(ch in allowed or ch.isspace() for ch in s):
                        tups.append((s, tid))
            except:
                continue

        return sorted(tups)

    def _get_token_probabilities_from_model(self, context_tokens, token_ids: List[int]) -> Dict[int, float]:
        """Get probabilities for specific tokens given context using model"""
        try:

            # convert context to tensor
            context_tensor = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

            probs, indices, past = get_probs_past(
                model=self.model,
                prev=context_tensor,
                past=None,
                device=DEVICE,
                top_p=0.95
            )

            # Ensure probs and indices are torch tensors (get_probs_past may return lists)
            if not isinstance(probs, torch.Tensor): # TODO just else
                probs = torch.tensor(probs, device=DEVICE, dtype=torch.float32)
            else:
                probs = probs.to(torch.float32).to(DEVICE)

            if not isinstance(indices, torch.Tensor): # TODO just else
                indices = torch.tensor(indices, device=DEVICE, dtype=torch.long)
            else:
                indices = indices.to(torch.long).to(DEVICE)

            # Extract probabilities for specific token_ids
            result_probs = {}
            for token_id in token_ids:
                matches = (indices == token_id).nonzero(as_tuple=True)[0]
                if matches.numel() > 0:
                    idx = matches[0].item()
                    result_probs[token_id] = float(probs[idx].item())
                else:
                    result_probs[token_id] = 1e-10  # Very small probability

            return result_probs

        except Exception as e:
            print(f"Error getting token probabilities: {e}")
            # Fallback to uniform probabilities
            prob_per_token = 1.0 / len(token_ids) if token_ids else 0.0
            return {token_id: prob_per_token for token_id in token_ids}


def _update_tree_from_verification(tree: BackCheckTree, result: BackCheckVerificationResult):
    """Update tree nodes based on verification results"""
    current_node = tree.root
    i = 0
    while i < len(result.correct_tokens):
        child_found = False
        for child in current_node.children:
            if child.token == result.correct_tokens[i]:
                child.known_correct = True
                current_node = child
                i += 1
                child_found = True
                break
        if not child_found:
            raise ValueError("Correct token not found in children during update")

    i = 0
    current_node = tree.root
    while i < len(result.visited_tokens):
        child_found = False
        for child in current_node.children:
            if child.token == result.visited_tokens[i]:
                current_node = child
                child_found = True
                i += 1
                break
        if not child_found:
            raise ValueError("Visited token not found in children during update")
    current_node.known_wrong = True


def _find_path_through_tree(tree: BackCheckTree) -> Optional[List[int]]:
    """Find path through tree following priority rules"""

    solution = []
    current_node = tree.root
    while len(TOKENIZER.decode(solution).strip()) < len(tree.stego_text.strip()):
        if len(current_node.children) == 0 or all(child.known_wrong for child in current_node.children):
            if not current_node.is_explored:
                # Explore this node to find possible continuations
                tree.explore_node(current_node)
                continue
        correct_children = [child for child in current_node.children if child.known_correct]
        if correct_children:
            current_node = correct_children[0]
            solution.append(current_node.token)
            continue
        greedy_children = [child for child in current_node.children if child.greedy_correct and not child.known_wrong]
        if greedy_children:
            current_node = greedy_children[0]
            solution.append(current_node.token)
            continue
        best_child = max(
            (child for child in current_node.children if not child.known_wrong),
            key=lambda c: c.likeliness,
            default=None
        )
        if best_child:
            current_node = best_child
            solution.append(current_node.token)
            continue

        if current_node.known_correct:
            raise ValueError("node known to be correct, but no possible path!")

        if current_node.parent is None:
            return None  # No valid path
        # if nothing is found, go back to parent
        current_node.known_wrong = True  # Mark node as wrong to stop further attempts
        solution.pop()
        current_node = current_node.parent


    print(solution)
    return solution


class BackCheckDecoder:
    """BackCheck decoder that integrates with existing decoding functions"""

    def __init__(self, model, tokenizer, context, device=DEVICE, **decode_params):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = device
        self.decode_params = decode_params

    def verify_path(self, token_path: List[int], last_verified_state : BlindDecodingState = None) -> BackCheckVerificationResult:
        """Verification function using existing try_blind_decoding"""
        try:
            # Use existing try_blind_decoding function
            success, final_state, last_idx = try_blind_decoding(
                token_sequence=token_path,
                last_verified_state=last_verified_state,
                model=self.model,
                device=self.device,
                block_size=self.decode_params.get('block_size', 32),
                top_p=self.decode_params.get('top_p', 0.95),
                blocks_per_interval=self.decode_params.get('blocks_per_interval', 4)
            )
            decoded_message = final_state.get_discovered_message_as_string()
            correct_tokens = token_path[:final_state.token_count] if final_state.token_count >= 0 else []

            visited_tokens = token_path[:last_idx] if last_idx >= 0 else []
            backcheck_count = final_state.backcheck_count
            print("Correct tokens:", correct_tokens)
            print("Visited tokens:", visited_tokens)

            return BackCheckVerificationResult(
                decoded_message=decoded_message,
                correct_tokens=correct_tokens,
                visited_tokens=visited_tokens,
                success=success,
                backcheck_count=backcheck_count,
                last_verified_state=final_state)


        except Exception as e:
            print(f"Error in BackCheck path verification: {e}")
            raise e

    def backcheck_decode_tree(self, tree: BackCheckTree, max_attempts: int = 100) -> Tuple[List[int], str, int]:
        """Main BackCheck algorithm implementation"""
        attempts = 0
        last_verified_state = init_decoding_state(context=self.context, initial_backcheck_count=self.decode_params.get('initial_backcheck_count', -1),random_seed=self.decode_params.get('random_seed', -1))
        while attempts < max_attempts:
            attempts += 1

            # Find path through tree following priority rules
            path_tokens = _find_path_through_tree(tree)

            if not path_tokens:
                print(f"BackCheck: No valid path found after {attempts} attempts")
                break

            # Verify the path using existing decoding
            result = self.verify_path(path_tokens, last_verified_state)
            last_verified_state = result.last_verified_state

            # Update tree based on verification results
            _update_tree_from_verification(tree, result)

            if result.success:
                print(f"✅ BackCheck succeeded after {attempts} attempts!")
                return path_tokens, result.decoded_message, last_verified_state.backcheck_count

            if attempts % 10 == 0:
                print(f"BackCheck: Attempt {attempts}, continuing search...")

        return None


def backcheck_decode_single_message(message: str, context: str, random_seed: int, backcheck_count: int) -> Tuple[
    str, int]:
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
            'top_p': 0.95,
            'random_seed': random_seed,
            'blocks_per_interval': 4,
            'initial_backcheck_count': backcheck_count,
            'get_probs_past_func': get_probs_past
        }

        # Create decoder with context
        context_tensor = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
        decoder = BackCheckDecoder(MODEL, TOKENIZER, context_tensor, DEVICE, **decoder_params)

        # Try BackCheck decoding
        result = decoder.backcheck_decode_tree(tree, max_attempts=500)

        if result:
            tokenization, decoded_message, backcheck_count = result
            print(f"✅ BackCheck successful: '{decoded_message}'")
            return decoded_message, backcheck_count
        else:
            raise ValueError("⚠️ BackCheck failed, falling back to linear approach")
    except Exception as e:
        print(f"Error in BackCheck decoding: {e}")
        raise e

# ============================================================================
# EXISTING CODE - PRESERVED WITH MINIMAL CHANGES
# ============================================================================

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

    def save_random_state(self):
        self.random_state_data = random.getstate()

    def restore_random_state(self):
        if self.random_state_data is not None:
            random.setstate(self.random_state_data)

    def add_discovered_block(self, block_bits: str, is_checkpoint: bool):
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
                       block_size: int = 32,
                       top_p: float = 0.95,
                       blocks_per_interval: int = 4) -> Tuple[bool, BlindDecodingState, int]:
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

            token_index = torch.where(indices == tokenID)[0]
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
        print(f"Exception during decoding: {e}")
        #raise e
        return False, last_verified_state, state.token_count


def init_decoding_state(context, initial_backcheck_count, random_seed):
    state = BlindDecodingState()
    state.past = None
    state.prev = context
    random.seed(random_seed)
    state.save_random_state()
    state.token_count = 0
    state.backcheck_count = initial_backcheck_count
    state.prev = context
    return state


def full_decode(context, messages, random_seed):
    """
    Main entry point for decoding - now with BackCheck integration

    Preserves the original function signature while adding BackCheck functionality
    """
    final_messages = []
    backcheck_count = 0
    rng = np.random.default_rng(random_seed)

    for i, message in enumerate(messages):
        print(f"\n🔍 Decoding message {i+1}/{len(messages)}")
        random_number = rng.integers(low=10**15, high=10**16)

        # Try BackCheck decoding first
        decoded_message, backcheck_count = backcheck_decode_single_message(message, context, random_number, backcheck_count)

        if decoded_message:
            final_messages.append(decoded_message)
        else:
            print(f"⚠️ All decoding approaches failed for message {i+1}")
            final_messages.append("")

    final_message = "".join(final_messages)
    return final_message
