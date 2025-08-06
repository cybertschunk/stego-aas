import random
import torch
from math import ceil
from .sparsamp_utils import func_mrn, dec2bin, get_lower_upper_bound, get_probs_past, MODEL, TOKENIZER, DEVICE, \
    string_to_utf8_binary, utf8_binary_to_string
import numpy as np

def process_message(message_text: str, block_size: int) -> str:
    """Process message to ensure length meets requirements"""
    message_bits = string_to_utf8_binary(message_text)
    if len(message_bits) % block_size != 0:
        padding_length = block_size - (len(message_bits) % block_size)
        message_bits = message_bits + '0' * padding_length
        print(f"Message length padded to {len(message_bits)} bits")
    return message_bits

def full_encode(context, message_text, random_seed):
    message_bits = process_message(message_text,32)
    final_messages = []
    i = 0
    tokenized_context = TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)
    rng = np.random.default_rng(random_seed)
    to_decode = message_bits
    while i < len(message_bits):
        random_number = rng.integers(low=10**15, high=10**16)
        min_tokens = min(100, len(to_decode)//7)
        generated_ids, encoded_messages = encode_spar(model=MODEL, context=tokenized_context, message_bits=to_decode,min_token_length=min_tokens,max_token_length=1000,random_seed=random_number, device=DEVICE)
        encoded_message = "".join(encoded_messages)
        m = TOKENIZER.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_messages.append(m)
        i += len(encoded_message)
        to_decode = to_decode[len(encoded_message):]
    return final_messages

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