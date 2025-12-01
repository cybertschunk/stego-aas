"""
BackCheck algorithm implementation for resolving tokenization ambiguity in steganographic decoding.

This module implements the BackCheck tree-based approach to handle cases where different
tokenizations of the same text can lead to different decoded messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from .constants import (
    BLOCK_SIZE, BLOCKS_PER_INTERVAL, TOP_P, MAX_BACKCHECK_ATTEMPTS,
    RANDOM_SEED_MIN, RANDOM_SEED_MAX
)
from .model_manager import get_model_manager
from .sparsamp_utils import get_probs_past

logger = logging.getLogger(__name__)


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
        while current.token != -1:  # stop at root node
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
    last_verified_state: 'BlindDecodingState' = None
    backcheck_count: int = -1


class BackCheckTree:
    """BackCheck tree for token ambiguity resolution"""

    def __init__(self, stego_text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device: str):
        self.stego_text = stego_text
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.root = BackCheckNode()  # Empty root node
        self.all_nodes: List[BackCheckNode] = []  # Keep track of all nodes

        # Build the initial tree with default tokenization
        self._build_initial_tree()

    def _build_initial_tree(self) -> None:
        """Build initial tree with default tokenization as greedy path"""
        # Get default (greedy) tokenization
        greedy_tokens = self.tokenizer.encode(self.stego_text)

        # Create nodes for default tokenization path
        current_node = self.root

        self._build_greedy_path(current_node, greedy_tokens)

    def _build_greedy_path(self, current_node: BackCheckNode, greedy_tokens: List[int]) -> None:
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

    def explore_node(self, node: BackCheckNode) -> None:
        """
        Explore function: determine all possible tokens that could follow after this node
        and calculate their probabilities using the model
        """

        # Get remaining text to match
        current_text = self.tokenizer.decode(node.get_path_tokens()).strip()
        remaining_text = self.stego_text.strip()[len(current_text):]
        if len(node.children) > 0:
            self.explore_model_based(node, remaining_text)
            node.is_explored = True
            return

        greedy_tokens = self.tokenizer.encode(remaining_text, add_special_tokens=False)
        self._build_greedy_path(node, greedy_tokens)

    def explore_model_based(self, node: BackCheckNode, remaining_text: str) -> None:
        """Explore node using model-based probability calculations"""
        # Find all possible tokens that could continue from this position
        possible_continuations = []
        # Use efficient approach to find valid continuations
        sorted_tokens = self._prepare_tokens_for_prefix(remaining_text)

        # Check each token to see if it could be a valid continuation
        for token_str, token_id in sorted_tokens:
            test_tokens = node.get_path_tokens() + [token_id]
            test_text = self.tokenizer.decode(test_tokens)
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

        return tups

    def _get_token_probabilities_from_model(self, context_tokens, token_ids: List[int]) -> Dict[int, float]:
        """Get probabilities for specific tokens given context using model"""
        try:

            # convert context to tensor
            context_tensor = torch.tensor(context_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            probs, indices, past = get_probs_past(
                model=self.model,
                prev=context_tensor,
                past=None,
                device=self.device,
                top_p=0.95
            )

            probs = probs.to(torch.float32).to(self.device)
            indices = indices.to(torch.long).to(self.device)

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
            logger.error(f"Error getting token probabilities: {e}")
            # Fallback to uniform probabilities
            prob_per_token = 1.0 / len(token_ids) if token_ids else 0.0
            return {token_id: prob_per_token for token_id in token_ids}


def _update_tree_from_verification(tree: BackCheckTree, result: BackCheckVerificationResult) -> None:
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


def _find_path_through_tree(tree: BackCheckTree, tokenizer: PreTrainedTokenizer) -> Optional[List[int]]:
    """Find path through tree following priority rules"""

    solution = []
    current_node = tree.root
    while len(tokenizer.decode(solution).strip()) < len(tree.stego_text.strip()):
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

    return solution


class BackCheckDecoder:
    """BackCheck decoder that integrates with existing decoding functions"""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, context: torch.Tensor, device: str, **decode_params):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = device
        self.decode_params = decode_params

    def verify_path(self, token_path: List[int], last_verified_state: 'BlindDecodingState' = None) -> BackCheckVerificationResult:
        """Verification function using existing try_blind_decoding"""
        # Import here to avoid circular imports
        from .decoding import try_blind_decoding

        try:
            # Use existing try_blind_decoding function
            success, final_state, last_idx = try_blind_decoding(
                token_sequence=token_path,
                last_verified_state=last_verified_state,
                model=self.model,
                device=self.device,
                block_size=self.decode_params.get('block_size', BLOCK_SIZE),
                top_p=self.decode_params.get('top_p', TOP_P),
                blocks_per_interval=self.decode_params.get('blocks_per_interval', BLOCKS_PER_INTERVAL)
            )
            decoded_message = final_state.get_discovered_message_as_string()
            correct_tokens = token_path[:final_state.token_count] if final_state.token_count >= 0 else []

            visited_tokens = token_path[:last_idx] if last_idx >= 0 else []
            backcheck_count = final_state.backcheck_count
            logger.debug(f"Correct tokens: {correct_tokens}")
            logger.debug(f"Visited tokens: {visited_tokens}")

            return BackCheckVerificationResult(
                decoded_message=decoded_message,
                correct_tokens=correct_tokens,
                visited_tokens=visited_tokens,
                success=success,
                backcheck_count=backcheck_count,
                last_verified_state=final_state)

        except Exception as e:
            logger.error(f"Error in BackCheck path verification: {e}")
            raise e

    def backcheck_decode_tree(self, tree: BackCheckTree, max_attempts: int = MAX_BACKCHECK_ATTEMPTS) -> Optional[Tuple[List[int], str, int, int]]:
        """
        Main BackCheck algorithm implementation

        Returns:
            Optional tuple of (path_tokens, decoded_message, backcheck_count, attempts) or None if failed
        """
        # Import here to avoid circular imports
        from .decoding import init_decoding_state

        attempts = 0
        last_verified_state = init_decoding_state(context=self.context, initial_backcheck_count=self.decode_params.get('initial_backcheck_count', -1), random_seed=self.decode_params.get('random_seed', -1))
        while attempts < max_attempts:
            attempts += 1

            # Find path through tree following priority rules
            path_tokens = _find_path_through_tree(tree, self.tokenizer)

            if not path_tokens:
                logger.warning(f"BackCheck: No valid path found after {attempts} attempts")
                break

            # Verify the path using existing decoding
            result = self.verify_path(path_tokens, last_verified_state)
            last_verified_state = result.last_verified_state

            # Update tree based on verification results
            _update_tree_from_verification(tree, result)

            if result.success:
                logger.info(f"BackCheck succeeded after {attempts} attempts")
                return path_tokens, result.decoded_message, last_verified_state.backcheck_count, attempts

            if attempts % 10 == 0:
                logger.debug(f"BackCheck: Attempt {attempts}, continuing search...")

        return None


def backcheck_decode_single_message(message: str, context: str, random_seed: int, backcheck_count: int) -> Tuple[str, int, int]:
    """
    Decode a single message using BackCheck algorithm

    Returns:
        Tuple of (decoded_message, backcheck_count, attempts)
    """
    model_manager = get_model_manager()
    try:
        # First try BackCheck approach
        logger.info(f"Trying BackCheck decoding for message length {len(message)}")

        # Build BackCheck tree
        tree = BackCheckTree(message, model_manager.tokenizer, model_manager.model, model_manager.device)

        # Set up decoder parameters
        decoder_params = {
            'block_size': BLOCK_SIZE,
            'top_p': TOP_P,
            'random_seed': random_seed,
            'blocks_per_interval': BLOCKS_PER_INTERVAL,
            'initial_backcheck_count': backcheck_count,
            'get_probs_past_func': get_probs_past
        }

        # Create decoder with context
        context_tensor = model_manager.tokenizer.encode(context, return_tensors='pt').to(model_manager.device)
        decoder = BackCheckDecoder(model_manager.model, model_manager.tokenizer, context_tensor, model_manager.device, **decoder_params)

        # Try BackCheck decoding
        result = decoder.backcheck_decode_tree(tree, max_attempts=MAX_BACKCHECK_ATTEMPTS)

        if result:
            tokenization, decoded_message, backcheck_count, attempts = result
            logger.info(f"BackCheck successful: '{decoded_message}' in {attempts} attempts")
            return decoded_message, backcheck_count, attempts
        else:
            raise ValueError("WARNING: BackCheck failed, falling back to linear approach")
    except Exception as e:
        logger.error(f"Error in BackCheck decoding: {e}")
        raise e
