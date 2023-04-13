import os
import torch
from transformers import AutoTokenizer, GPTJForCausalLM, pipeline


def model_fn(model_dir):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", revision="float16", torch_dtype=torch.float16)

    print('Model downloaded and loaded into memory...')

    if torch.cuda.is_available():
        device = 0
    else:
        device = -1

    generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    return generation