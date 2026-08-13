"""
Super simple LLM from scratch — tiny character-level GPT.
Train on a tiny built-in corpus, then generate text.

  pip install torch
  python llm.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tiny dataset (swap for your own text) ---
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
BLOCK_SIZE = 64      # context length
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
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


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
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def loss(self, idx):
        logits = self(idx)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train(model, data):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")


def main():
    stoi, itos = build_vocab(CORPUS)
    data = encode(CORPUS, stoi).to(DEVICE)

    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=BLOCK_SIZE,
        n_embd=EMBED_DIM,
        n_head=NUM_HEADS,
        n_layer=NUM_LAYERS,
    ).to(DEVICE)

    print(f"device: {DEVICE} | params: {sum(p.numel() for p in model.parameters()):,}")
    train(model, data)

    model.eval()
    start = "to be"
    start_ids = encode(start, stoi).unsqueeze(0).to(DEVICE)
    out = model.generate(start_ids, GENERATE_LEN, temperature=0.8)
    print("\n--- generated ---")
    print(decode(out[0].tolist(), itos))


if __name__ == "__main__":
    main()