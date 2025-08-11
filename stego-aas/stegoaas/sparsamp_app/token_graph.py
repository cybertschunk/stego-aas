from __future__ import annotations        # no effect in 3.8 but safe
import sys
from typing import Dict, List, Tuple, Iterable

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

Edge = Tuple[int, int]                    # (next_char_index, token_id)
TokenGraph = Dict[int, List[Edge]]        # char_index -> list of edges


from bisect import bisect_left, bisect_right

def _prepare_tokens(tokenizer, text):
    allowed = set(text)                       # every Unicode character in the text
    tups = []                                 # (decoded_string, token_id)

    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and all(ch in allowed or ch.isspace() for ch in s):
            tups.append((s, tid))             # keep only UTF-8 chars you know occur

    return sorted(tups)                       # lexicographic order

def _candidates(sorted_tokens, prefix):
    """
    Return the slice of tokens whose decoded string starts with `prefix`.
    """
    lo  = bisect_left(sorted_tokens, (prefix, -1))
    hi  = bisect_right(sorted_tokens, (prefix + '\uffff', float('inf')))
    return sorted_tokens[lo:hi]


def build_token_graph(text, tokenizer):
    tokens_sorted = _prepare_tokens(tokenizer, text)
    n        = len(text)
    graph    = {i: [] for i in range(n + 1)}
    reachable = [False]*(n + 1); reachable[0] = True

    for i in range(n):
        if not reachable[i]:
            continue

        # longest prefix we ever need to query = longest token
        for L in range(1, min(12, n - i) + 1):          # GPT-2 max token length ≈12
            prefix = text[i:i+L]
            for tok_str, tid in _candidates(tokens_sorted, prefix):
                if text.startswith(tok_str, i):
                    j = i + len(tok_str)
                    graph[i].append((j, tid))
                    reachable[j] = True
            # stop early if no token starts with this longer prefix
            if not _candidates(tokens_sorted, prefix):
                break
    return graph




def all_tokenizations(
    graph: TokenGraph,
    end_pos: int
) -> Iterable[List[int]]:
    """
    Depth-first generator of complete token-id paths that reach end_pos.
    """
    stack: List[Tuple[List[int], int]] = [([], 0)]    # (path, current_pos)
    while stack:
        tokens, pos = stack.pop()
        if pos == end_pos:
            yield tokens
            continue
        for nxt_pos, tok_id in graph.get(pos, []):
            stack.append((tokens + [tok_id], nxt_pos))



def best_next_tokenization(
    full_text: str,
    graph: TokenGraph,
    prefix_tokens: List[int],
    negative_next_lists: List[List[int]],
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
) -> List[int]:
    """
    Parameters
    ----------
    full_text : str
        Complete plain-text string (no special tokens).
    graph : Graph
        Output of build_token_graph(full_text, ...).
    prefix_tokens : List[int]
        *Correct* token-id sequence covering full_text[0 : k].
    negative_next_lists : List[List[int]]
        Each entry is a token-id list that *must NOT* be chosen as
        the next-segment tokenization.  All lists start at char offset k
        and cover the same upcoming span (length ≥1 chars).
    model / tokenizer
        Pre-loaded GPT-2 objects.

    Returns
    -------
    List[int]
        Highest-probability token-id sequence that
          • begins at char offset k (=len(decoded(prefix_tokens))),
          • ends at char offset m (m > k) *before* any later negatives start,
          • is NOT equal to any list in `negative_next_lists`.
    """

    k = len(tokenizer.decode(prefix_tokens, skip_special_tokens=True))
    text_len = len(full_text)

    # ------------ helper: Viterbi forward pass -----------------
    NEG_INF = -1e9
    best_lp: List[float] = [NEG_INF]*(text_len + 1)
    best_prev: List[Tuple[int, int] | None] = [None]*(text_len + 1)
    best_lp[k] = 0.0                      # prefix is assumed correct

    # cache model outputs for identical contexts
    logprob_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    for i in range(k, text_len):
        if best_lp[i] == NEG_INF:
            continue

        # ---- get log-probs for next token given current best prefix ----
        prefix_up_to_i = _reconstruct(best_prev, i, prefix_tokens)
        ctx_key = tuple(prefix_up_to_i)
        if ctx_key in logprob_cache:
            log_probs = logprob_cache[ctx_key]
        else:
            with torch.no_grad():
                inp = torch.tensor([ctx_key], dtype=torch.long)
                logits = model(inp).logits[0, -1]          # last position
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprob_cache[ctx_key] = log_probs

        # ---- expand outgoing one-token edges ----
        for j, tok_id in graph[i]:
            lp = best_lp[i] + log_probs[tok_id].item()
            if lp > best_lp[j]:
                best_lp[j]   = lp
                best_prev[j] = (i, tok_id)

    # --------- gather all candidate paths that avoid the negatives ---------
    negative_sets = {tuple(seq) for seq in negative_next_lists}

    candidates: List[Tuple[float, List[int]]] = []
    for end_pos in range(k + 1, text_len + 1):
        toks = _reconstruct(best_prev, end_pos, prefix_tokens)
        next_chunk = toks[len(prefix_tokens):]            # slice after prefix
        if next_chunk and tuple(next_chunk) not in negative_sets:
            candidates.append((best_lp[end_pos], next_chunk))

    if not candidates:
        raise ValueError("No admissible tokenization found.")

    # return highest-probability admissible path
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------- util: back-pointer reconstruction -----------------
def _reconstruct(
    best_prev: List[Tuple[int, int] | None],
    pos: int,
    prefix_tokens: List[int]
) -> List[int]:
    toks = []
    while pos >= 0 and best_prev[pos] is not None:
        prev_pos, tok_id = best_prev[pos]
        toks.append(tok_id)
        pos = prev_pos
    toks.extend(reversed(prefix_tokens))
    return list(reversed(toks))