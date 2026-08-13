import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tiny dataset ---
CORPUS = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
    "and by opposing end them to die to sleep "
    "no more and by a sleep to say we end "
    "the heartache and the thousand natural shocks "
    "that flesh is heir to tis a consummation "
    "devoutly to be wished to die to sleep to sleep "
    "perchance to dream ay theres the rub "
) * 20


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
LR = 3e-3
STEPS = 800
GENERATE_LEN = 200

def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoic[c] for c in text], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos[i] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.trill(torch.ones(block_size, block_size)))

    