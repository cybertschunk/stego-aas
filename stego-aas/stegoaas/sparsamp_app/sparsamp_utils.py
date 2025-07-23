# This file is adapted from:
# Wang, Y. (2025). Artifact for "SparSamp: Efficient Provably Secure Steganography Based on Sparse Sampling".
# Zenodo. https://doi.org/10.5281/zenodo.15025436
#
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See https://creativecommons.org/licenses/by/4.0/ for details.

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOKENIZER = None
DEVICE = None


def get_lower_upper_bound(cumulative_probs, v):
    # 计算下界和上界
    lower_bound = cumulative_probs[v - 1] if v > 0 else torch.tensor(0)
    upper_bound = cumulative_probs[v] if v < len(cumulative_probs) - 1 else torch.tensor(1)
    SE = [lower_bound.item(), upper_bound.item()]
    return SE


def func_mrn(k_m, n_m, r):
    result = ((k_m / n_m) + r)
    if result >= 1:
        # print(f"result after subtraction: {result}")
        result = result - 1

    return result


def dec2bin(km, lm):
    # 将 km 转换为二进制字符串，去掉 '0b' 前缀
    bin_str = bin(km)[2:]

    # 使用 zfill 填充到 lm 位，保证长度为 lm
    return bin_str.zfill(lm)


def load_model():
    global MODEL, TOKENIZER, DEVICE
    MODEL = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    TOKENIZER = AutoTokenizer.from_pretrained("openai-community/gpt2")
    DEVICE = torch.device("cpu")
    MODEL.to(DEVICE)
    MODEL.eval()


def limit_past(past):
    if past is None:
        return None
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -1022:]
            # past[i][j] = past[i][j][:, :, -256:]
    return past

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += int(bit) * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def get_probs_past(model,
                   prev=None,
                   past=None,
                   device='cuda',
                   top_p=1.0):
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



def get_logits(model, input_ids):
    model_output = model(input_ids)

    logits_list = model_output.logits
    return logits_list


def process_logits_to_probs(logits_list, logits_index, top_p):
    # logits_index 为负数，从-1开始
    logits = logits_list[0, logits_index, :]
    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    probs = F.softmax(logits, dim=-1)

    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        k = (cum_probs > top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices


def find_nearest(anum: float, probs: List[float]) -> int:
    # Returns index_idx (index of indices)
    up = len(probs) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index_idx = int((up + bottom) / 2)
        if probs[index_idx] < anum:
            up = index_idx
        elif probs[index_idx] > anum:
            bottom = index_idx
        else:
            return index_idx
    if up - bottom == 1:
        if probs[bottom] - anum < anum - probs[up]:
            index_idx = bottom
        else:
            index_idx = up
    return index_idx


def load_context(file):
    import pandas as pd
    df = pd.read_excel(file)
    context_list = df['context'].tolist()
    return context_list


def get_bits_length_from_list(encoded_messages):
    # 统计嵌入的总比特数和生成的token数
    total_encoded_bits_length = 0
    for encoded_message in encoded_messages:
        cur_encoded_bits_length = len(encoded_message)
        total_encoded_bits_length += cur_encoded_bits_length
    return total_encoded_bits_length


from math import ceil, floor


def custom_round(type, x):
    if type == "round":
        return round(x)
    elif type == "ceil":
        return ceil(x)
    elif type == "floor":
        return floor(x)
    else:
        raise ValueError("Invalid round_type. Use 'round', 'ceil', 'floor'.")


def string_to_utf8_binary(s):
    return ''.join(f"{byte:08b}" for byte in s.encode('utf-8'))
