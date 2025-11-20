# This file is adapted from:
# Wang, Y. (2025). Artifact for "SparSamp: Efficient Provably Secure Steganography Based on Sparse Sampling".
# Zenodo. https://doi.org/10.5281/zenodo.15025436
#
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See https://creativecommons.org/licenses/by/4.0/ for details.

import torch
import torch.nn.functional as F

from .constants import CONTEXT_WINDOW_SIZE


def get_lower_upper_bound(cumulative_probs, v):
    """Calculate lower and upper bounds for a token in the cumulative probability distribution."""
    lower_bound = cumulative_probs[v - 1] if v > 0 else torch.tensor(0)
    upper_bound = cumulative_probs[v] if v < len(cumulative_probs) - 1 else torch.tensor(1)
    SE = [lower_bound.item(), upper_bound.item()]
    return SE


def func_mrn(k_m, n_m, r):
    """
    Calculate the MRN (Modified Random Number) for encoding.
    Normalizes the result to [0, 1) by subtracting 1 if >= 1.
    """
    result = ((k_m / n_m) + r)
    if result >= 1:
        result = result - 1
    return result


def dec2bin(km, lm):
    """
    Convert decimal to binary string.

    Args:
        km: Decimal number to convert
        lm: Length of the binary string (will be zero-padded if necessary)

    Returns:
        Binary string of length lm
    """
    bin_str = bin(km)[2:]  # Convert to binary and remove '0b' prefix
    return bin_str.zfill(lm)  # Zero-pad to ensure length is lm


def limit_past(past):
    """
    Limit the past key-value cache to the maximum context window size.

    Args:
        past: Past key-value cache from the transformer model

    Returns:
        Truncated past cache limited to CONTEXT_WINDOW_SIZE
    """
    if past is None:
        return None
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -CONTEXT_WINDOW_SIZE:]
    return past


def get_probs_past(model,
                   prev=None,
                   past=None,
                   device='cuda',
                   top_p=1.0):
    """
    Get next token probabilities from the language model using past key-value cache.

    Args:
        model: The language model
        prev: Previous token IDs
        past: Past key-value cache from previous generation steps
        device: Device to run computation on ('cuda' or 'cpu')
        top_p: Top-p (nucleus) sampling threshold (0 < top_p <= 1.0)

    Returns:
        Tuple of (probabilities, token indices, updated past cache)
    """
    if past is not None:
        past = limit_past(past)
    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values

    logits = model_output.logits[0,-1,:].to(device)
    logits,indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    probs = F.softmax(logits, dim=-1)

    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        k = (cum_probs > top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices, past


def string_to_utf8_binary(s):
    """
    Convert a string to its UTF-8 binary representation.

    Args:
        s: Input string

    Returns:
        Binary string where each character is represented as 8 bits
    """
    return ''.join(f"{byte:08b}" for byte in s.encode('utf-8'))


def utf8_binary_to_string(bstr):
    """
    Convert a UTF-8 binary string back to a regular string.

    Args:
        bstr: Binary string (must be a multiple of 8 bits)

    Returns:
        Decoded UTF-8 string
    """
    # Split the binary string into 8-bit bytes
    bytes_list = [int(bstr[i:i+8], 2) for i in range(0, len(bstr), 8)]
    # Create bytes object
    byte_obj = bytes(bytes_list)
    # Decode as UTF-8
    return byte_obj.decode('utf-8')
