from __future__ import annotations

import random
from math import ceil

import torch
import numpy as np

from .sparsamp_utils import get_probs_past, TOKENIZER, get_lower_upper_bound, dec2bin, MODEL, DEVICE, \
    utf8_binary_to_string

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set


import tiktoken


# ---------------------------
# Blackbox decoder interface
# ---------------------------

def try_decoding(tokens: List[int]) -> Tuple[bool, int]:
    """
    Blackbox function. Implement externally.

    Returns:
        ok: True if 'tokens' represents the correct original tokenization.
        last_ok_byte: a UTF-8 byte offset into the original text that is confirmed correct.
                      If ok is True, this should equal len(x_bytes). If False, it should be
                      the maximal correct prefix boundary.
    """
    context = torch.tensor(tokenized_context[-1022:], device=device, dtype=torch.long)

    random.seed(random_seed)
    message = []
    n_m = 2 ** block_size
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []
    tokens_correct = []
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

        if n_m == 0:
            return False, tokens_correct, message
        if n_m == 1:
            tokens_correct.append(tokenID)
            count = len(temp0_arr) - 2
            k_m = temp0_arr[count + 1]
            while count >= 0:
                n_m_new = n_m_arr[count]
                k_m = temp0_arr[count] + ((k_m + n_m_new) % n_m_new)
                count -= 1
            k_m = (k_m + 2 ** block_size) % 2 ** block_size
            message.append(dec2bin(k_m, block_size))
            return True, tokens_correct, message
        prev = torch.tensor([tokenID], device=device, dtype=torch.long).unsqueeze(0)
    return None


# ---------------------------
# Token trie over bytes
# ---------------------------

class TokenTrieNode:
    __slots__ = ("children", "terminal_tokens")

    def __init__(self):
        self.children: Dict[int, TokenTrieNode] = {}
        self.terminal_tokens: List[int] = []

class TokenTrie:
    def __init__(self):
        self.root = TokenTrieNode()

    def insert(self, token_bytes: bytes, token_id: int):
        node = self.root
        for b in token_bytes:
            node = node.children.setdefault(b, TokenTrieNode())
        node.terminal_tokens.append(token_id)

    def iter_matches_at(self, data: bytes, pos: int) -> Iterable[int]:
        """
        Yield all token_ids whose decoded bytes match data starting at pos.
        Includes overlapping terminals along the walk (shorter to longer).
        """
        node = self.root
        i = pos
        while i < len(data):
            b = data[i]
            if b not in node.children:
                break
            node = node.children[b]
            if node.terminal_tokens:
                for tid in node.terminal_tokens:
                    yield tid
            i += 1


# ---------------------------
# Search node
# ---------------------------

@dataclass
class TokenNode:
    token_id: int
    text_bytes: bytes
    start: int
    end: int
    children: List["TokenNode"] = field(default_factory=list)
    assertedCorrect: bool = False
    assertedWrong: bool = False
    greedyCorrect: bool = False
    likeliness: float = -1.0

    def __hash__(self):
        return hash((self.token_id, self.start, self.end))


# ---------------------------
# Builder and search
# ---------------------------

class GPT2Recover:
    def __init__(self, text: str):
        self.text = text
        self.x_bytes = text.encode("utf-8")
        self.enc = tiktoken.encoding_for_model("gpt2")

        # Precompute token -> bytes and build trie
        self.token_bytes: Dict[int, bytes] = {}
        self.trie = TokenTrie()
        self._build_trie()

        # Greedy path for prioritization (token ids and per-token byte spans)
        self.greedy_tokens: List[int] = self.enc.encode(self.text)
        self.greedy_positions: Dict[int, int] = {}  # byte_pos -> token_id
        self._index_greedy_positions()

        # Flags / pruning
        self.wrong_edges: Set[Tuple[int, int]] = set()  # (start_byte, token_id)
        self.correct_until: int = 0  # max byte offset known to be correct

        # Cache nodes by (start, token_id) to avoid duplicates
        self.node_cache: Dict[Tuple[int, int], TokenNode] = {}

    def _build_trie(self):
        # token ids range: 0..enc.max_token_value (inclusive)
        n_vocab = self.enc.max_token_value + 1
        for tid in range(n_vocab):
            try:
                tb = self.enc.decode_single_token_bytes(tid)
            except KeyError:
                # Non-mergeable or invalid token id; skip
                continue
            if not tb:
                continue
            self.token_bytes[tid] = tb
            self.trie.insert(tb, tid)

    def _index_greedy_positions(self):
        # Map starting byte offset of each greedy token to its tid
        pos = 0
        for tid in self.greedy_tokens:
            tb = self.enc.decode_single_token_bytes(tid)
            self.greedy_positions[pos] = tid
            pos += len(tb)

    def _make_node(self, start: int, token_id: int) -> TokenNode:
        key = (start, token_id)
        if key in self.node_cache:
            return self.node_cache[key]
        tb = self.token_bytes[token_id]
        node = TokenNode(token_id=token_id, text_bytes=tb, start=start, end=start + len(tb))
        # Greedy flag if this token matches the greedy token at this position
        node.greedyCorrect = (self.greedy_positions.get(start) == token_id)
        # Likeliness: prioritize greedy first, then longer tokens
        node.likeliness = (1.0 if node.greedyCorrect else 0.0) + 0.0001 * len(tb)
        self.node_cache[key] = node
        return node

    def _children_at(self, pos: int) -> List[TokenNode]:
        # Enumerate all tokens matching at pos
        cand = []
        for tid in self.trie.iter_matches_at(self.x_bytes, pos):
            if (pos, tid) in self.wrong_edges:
                continue
            node = self._make_node(pos, tid)
            if node.end <= len(self.x_bytes):  # safety
                cand.append(node)
        # Sort: greedy first, then longer token first
        cand.sort(key=lambda n: (not n.greedyCorrect, -len(n.text_bytes)))
        return cand

    def _constrained_greedy_completion(self, start_pos: int) -> Optional[List[int]]:
        """
        Complete from start_pos using greedy tokens, but skip any (pos, tid) in wrong_edges.
        If the greedy token is blocked, fall back to “longest allowed match” at that pos.
        Returns a token id list or None if impossible under current constraints.
        """
        pos = start_pos
        seq: List[int] = []
        while pos < len(self.x_bytes):
            greedy_tid = self.greedy_positions.get(pos)
            chosen_tid = None
            # Try greedy first if allowed
            if greedy_tid is not None and (pos, greedy_tid) not in self.wrong_edges:
                chosen_tid = greedy_tid
            else:
                # Choose the longest allowed match
                best: Optional[Tuple[int, int]] = None  # (len, tid)
                for tid in self.trie.iter_matches_at(self.x_bytes, pos):
                    if (pos, tid) in self.wrong_edges:
                        continue
                    tb = self.token_bytes[tid]
                    if best is None or len(tb) > best[0]:
                        best = (len(tb), tid)
                if best is None:
                    return None
                chosen_tid = best[1]
            seq.append(chosen_tid)
            pos += len(self.token_bytes[chosen_tid])
        return seq

    def recover(self) -> List[int]:
        """
        Main entry: DFS guided by greedy and by pruning from try_decoding feedback.
        Returns the final correct tokenization (token ids).
        """
        stack: List[int] = [0]  # stack of byte positions (path implicit via parents map)
        parents: Dict[int, Tuple[Optional[int], Optional[TokenNode]]] = {0: (None, None)}
        # next_child_idx maps position -> which child index to try next
        next_child_idx: Dict[int, int] = {}

        # Precompute children list lazily and cache per position for stable ordering
        children_cache: Dict[int, List[TokenNode]] = {}

        def path_tokens_to(pos: int) -> List[int]:
            # Reconstruct tokens from parents up to pos
            seq: List[int] = []
            cur = pos
            while True:
                ppos, pnode = parents[cur]
                if ppos is None:
                    break
                assert pnode is not None
                seq.append(pnode.token_id)
                cur = ppos
            seq.reverse()
            return seq

        n_bytes = len(self.x_bytes)

        while stack:
            pos = stack[-1]

            # If we reached end, verify full path
            if pos == n_bytes:
                seq = path_tokens_to(pos)
                ok, last_ok_byte = try_decoding(seq)
                if ok:
                    # Mark nodes on the path as assertedCorrect
                    cur = pos
                    while True:
                        ppos, pnode = parents[cur]
                        if ppos is None or pnode is None:
                            break
                        pnode.assertedCorrect = True
                        cur = ppos
                    return seq

                # Mark correctness/wrongness and attempt constrained greedy completion
                self.correct_until = max(self.correct_until, last_ok_byte)
                # Mark wrong edges along the suffix after last_ok_byte
                # Walk back from end to find boundary
                cur = pos
                while True:
                    ppos, pnode = parents[cur]
                    if ppos is None or pnode is None:
                        break
                    if pnode.start >= last_ok_byte:
                        pnode.assertedWrong = True
                        self.wrong_edges.add((pnode.start, pnode.token_id))
                    else:
                        pnode.assertedCorrect = True
                    cur = ppos

                # Attempt a constrained greedy completion from last_ok_byte
                completion = self._constrained_greedy_completion(last_ok_byte)
                if completion is not None:
                    # Build a full sequence = confirmed prefix + completion
                    prefix = path_tokens_to(last_ok_byte)
                    seq2 = prefix + completion
                    ok2, last_ok2 = try_decoding(seq2)
                    if ok2:
                        # Success; we can reconstruct nodes for seq2 if needed
                        return seq2
                    # Update wrong set from last_ok2 as well
                    self.correct_until = max(self.correct_until, last_ok2)
                    # The constrained greedy either violated constraints or blackbox disagreed;
                    # continue DFS exploration from boundary
                # Backtrack
                stack.pop()
                continue

            # Generate children at this position
            if pos not in children_cache:
                children_cache[pos] = self._children_at(pos)
                next_child_idx[pos] = 0

            kids = children_cache[pos]
            idx = next_child_idx[pos]

            # Exhausted children -> backtrack
            if idx >= len(kids):
                stack.pop()
                continue

            # Take next child
            node = kids[idx]
            next_child_idx[pos] += 1

            # Skip edges known wrong
            if (node.start, node.token_id) in self.wrong_edges:
                continue

            # Advance
            next_pos = node.end
            if next_pos not in parents:
                parents[next_pos] = (pos, node)
                stack.append(next_pos)
            else:
                # Already visited this byte pos via some other path; still push to explore deeper
                # but keep only the first parent to limit memory; exploration still works by pos frontier.
                stack.append(next_pos)

        raise RuntimeError("No valid tokenization found under given constraints.")


# ---------------------------
# Convenience function
# ---------------------------

def recover_gpt2_tokens(text: str) -> List[int]:
    """
    Recover the original GPT-2 tokenization of 'text', guided by a SparSamp-style try_decoding.
    """
    engine = GPT2Recover(text)
    return engine.recover()





def remove_checkpoint_bits(bits, block_size, blocks_per_interval):
    """Remove checkpoint bytes from bitstring (for post-processing decoded bits)."""
    out = []
    n_blocks = len(bits) // block_size
    for i in range(n_blocks):
        block_bits = bits[i * block_size:(i + 1) * block_size]
        # Remove checkpoint bits if needed
        if (i + 1) % blocks_per_interval == 0:
            out.append(block_bits[:block_size - 8])
        else:
            out.append(block_bits)
    return ''.join(out)


def full_decode(
        context_text: str,
        messages: list,
        random_seed,
        block_size=32,
        blocks_per_interval=4,
        top_p=1.0,
):
    """
    Decodes a list of covertext messages, each generated by encode_spar.
    After decoding, removes checkpoint bits and reconstructs the UTF-8 plaintext.
    """
    bitstream = []
    rng = np.random.default_rng(random_seed)
    context_tokens= TOKENIZER.encode(context_text, return_tensors='pt').to(DEVICE)
    for message in messages:
        random_number = rng.integers(low=10 ** 15, high=10 ** 16)
        decoded_message = decode_spar(
            model=MODEL, message=message, tokenized_context=context_tokens, device=DEVICE,
            block_size=block_size, top_p=top_p, random_seed=random_number)
        bitstream.extend(decoded_message)
    bits = ''.join(bitstream)
    clean_bits = remove_checkpoint_bits(bits, block_size, blocks_per_interval)
    text = utf8_binary_to_string(clean_bits).rstrip('\x00')
    return text
