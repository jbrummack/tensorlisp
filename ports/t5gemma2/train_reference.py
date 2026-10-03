"""PyTorch reference for the LoRA training of ports/t5gemma2/train.ss (CPU, float32).

    python train_reference.py MODEL.safetensors OUT_DIR [--steps N] [--lr LR]

transformers can't load the gated google/t5gemma-2-270m-270m without its config, so this
re-implements the text encoder-decoder (Gemma 3 blocks, merged self/cross attention in the
decoder) from the checkpoint's tensors, plus PEFT-style LoRA on every linear layer:
y = W x + (alpha / r) B (A x). It exports, as .npy (numpy order, same names as the program's
parameters lora.<module>.A / .B):

    batch/<input>.npy        the entry's inputs
    init/<param>.npy         initial adapters (A uniform(1/sqrt(in)), B small random so that
                             every gradient is non-zero)
    grads/<param>.npy        d loss / d param at init
    loss.npy                 loss at init
    nll.npy                  per-token -log p at init
    trajectory/loss.npy      losses of the AdamW steps (same batch every step)
    trajectory/<param>.npy   parameters after those steps
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=16.0)
ap.add_argument("--src", type=int, default=12, help="encoder tokens")
ap.add_argument("--tgt", type=int, default=8, help="decoder tokens")
ap.add_argument("--f64", action="store_true", help="compute in float64 (to measure float32 noise)")
ap.add_argument("--f16-weights", action="store_true",
                help="round the matrices through f16 like the GGUF (norm weights stay f32)")
args = ap.parse_args()

if args.f64:
    torch.set_default_dtype(torch.float64)
DT = torch.get_default_dtype()
torch.manual_seed(0)
rng = np.random.default_rng(0)
H, NH, HD, NL, EPS, VOCAB = 640, 4, 256, 18, 1e-6, 262144
assert args.src < 256 and args.tgt < 256, "sliding windows are not implemented here"

sf = safe_open(args.model, "pt")
_cache = {}


def W(name):
    if name not in _cache:
        w = sf.get_tensor("model." + name).float()  # bf16 -> f32 exactly
        if args.f16_weights and w.dim() == 2:
            w = w.half().float()
        _cache[name] = w.to(DT)
    return _cache[name]


def sliding(i):
    return (i + 1) % 6 != 0


def rms(x, name):
    # Gemma: x / rms(x) * (1 + w)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + W(name + ".weight"))


def rope(x, pos, i):
    base, scale = (10000.0, 1.0) if sliding(i) else (1000000.0, 0.125)
    inv = base ** (-torch.arange(0, HD, 2).to(DT) / HD)
    ang = (pos.to(DT) * scale)[:, None] * inv[None, :]          # [T, HD/2]
    cos, sin = ang.cos(), ang.sin()
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)             # broadcast over heads
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


# --- LoRA
TARGETS = [("self_attn.q_proj", 640, 1024), ("self_attn.k_proj", 640, 256), ("self_attn.v_proj", 640, 256),
           ("self_attn.o_proj", 1024, 640), ("mlp.gate_proj", 640, 2048), ("mlp.up_proj", 640, 2048),
           ("mlp.down_proj", 2048, 640)]
lora = {}
for stack in ("encoder", "decoder"):
    for i in range(NL):
        for suffix, fin, fout in TARGETS:
            name = f"{stack}.layers.{i}.{suffix}"
            A = torch.tensor(rng.uniform(-1, 1, (args.rank, fin)) / math.sqrt(fin), dtype=DT)
            B = torch.tensor(rng.normal(0, 0.02, (fout, args.rank)), dtype=DT)
            lora[name] = (A.requires_grad_(), B.requires_grad_())
scale_lora = args.alpha / args.rank


def linear(x, name):
    y = x @ W(name + ".weight").T
    if name in lora:
        A, B = lora[name]
        y = y + scale_lora * ((x @ A.T) @ B.T)
    return y


def mlp(x, p):
    return linear(F.gelu(linear(x, p + ".mlp.gate_proj"), approximate="tanh") * linear(x, p + ".mlp.up_proj"), p + ".mlp.down_proj")


def attention(q, k, v, mask):
    # q [T, NH, HD], k/v [S, HD] (one kv head), mask [T, S] additive
    s = torch.einsum("tnd,sd->nts", q, k) / math.sqrt(HD) + mask[None]
    return torch.einsum("nts,sd->tnd", s.softmax(-1), v).reshape(q.shape[0], NH * HD)


NEG = -1e30


def encoder(ids, pos, pad):
    x = W("encoder.embed_tokens.weight")[ids] * 25.25
    L = ids.shape[0]
    mask = torch.where(pad[None, :] > 0, 0.0, NEG).expand(L, L)
    for i in range(NL):
        p = f"encoder.layers.{i}"
        h = rms(x, p + ".pre_self_attn_layernorm")
        q = rope(rms(linear(h, p + ".self_attn.q_proj").reshape(L, NH, HD), p + ".self_attn.q_norm"), pos, i)
        k = rope(rms(linear(h, p + ".self_attn.k_proj"), p + ".self_attn.k_norm"), pos, i)
        v = linear(h, p + ".self_attn.v_proj")
        x = x + rms(linear(attention(q, k, v, mask), p + ".self_attn.o_proj"), p + ".post_self_attn_layernorm")
        x = x + rms(mlp(rms(x, p + ".pre_feedforward_layernorm"), p), p + ".post_feedforward_layernorm")
    return rms(x, "encoder.norm")


def decoder(tokens, dpos, memory, pad):
    x = W("encoder.embed_tokens.weight")[tokens] * math.sqrt(H)
    T, M = tokens.shape[0], memory.shape[0]
    causal = torch.where(torch.arange(T)[None, :] <= torch.arange(T)[:, None], 0.0, NEG)
    cross = torch.where(pad[None, :] > 0, 0.0, NEG).expand(T, M)
    mask = torch.cat([causal, cross], 1)
    for i in range(NL):
        p = f"decoder.layers.{i}"
        h = rms(x, p + ".pre_self_attn_layernorm")
        q = rope(rms(linear(h, p + ".self_attn.q_proj").reshape(T, NH, HD), p + ".self_attn.q_norm"), dpos, i)
        k_self = rope(rms(linear(h, p + ".self_attn.k_proj"), p + ".self_attn.k_norm"), dpos, i)
        k_mem = rms(linear(memory, p + ".self_attn.k_proj"), p + ".self_attn.k_norm")
        k = torch.cat([k_self, k_mem], 0)
        v = torch.cat([linear(h, p + ".self_attn.v_proj"), linear(memory, p + ".self_attn.v_proj")], 0)
        x = x + rms(linear(attention(q, k, v, mask), p + ".self_attn.o_proj"), p + ".post_self_attn_layernorm")
        x = x + rms(mlp(rms(x, p + ".pre_feedforward_layernorm"), p), p + ".post_feedforward_layernorm")
    return rms(x, "decoder.norm")


def loss_fn(ids, pos, pad, tokens, dpos, targets, weights):
    memory = encoder(ids, pos, pad)
    hidden = decoder(tokens, dpos, memory, pad)
    logits = hidden @ W("encoder.embed_tokens.weight").T
    nll = -torch.log_softmax(logits, -1)[torch.arange(tokens.shape[0]), targets]
    return (nll * weights).sum(), nll


# --- batch: random token ids, the first half of the targets is "prompt" (weight 0)
L, T = args.src, args.tgt
ids = torch.tensor(rng.integers(1000, 20000, L))
pad = torch.ones(L, dtype=DT)
pad[-2:] = 0.0                                   # two padding positions
pos = torch.arange(L)
tokens = torch.tensor(np.concatenate([[2], rng.integers(1000, 20000, T - 1)]))
dpos = torch.arange(T)
targets = torch.tensor(rng.integers(1000, 20000, T))
weights = torch.zeros(T, dtype=DT)
weights[T // 2:] = 1.0
weights = weights / weights.sum()


def save(path, t):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(t.detach().numpy().astype(np.float32)))


for name, t in dict(ids=ids, gather=torch.arange(L), pos=pos, mask=pad, tokens=tokens, dpos=dpos,
                    targets=targets, weights=weights).items():
    save(f"{args.out}/batch/{name}.npy", t.float()[None, :])
for name, (A, B) in lora.items():
    save(f"{args.out}/init/lora.{name}.A.npy", A)
    save(f"{args.out}/init/lora.{name}.B.npy", B)

loss, nll = loss_fn(ids, pos, pad, tokens, dpos, targets, weights)
loss.backward()
save(f"{args.out}/loss.npy", loss.reshape(1))
save(f"{args.out}/nll.npy", nll)
for name, (A, B) in lora.items():
    save(f"{args.out}/grads/lora.{name}.A.npy", A.grad)
    save(f"{args.out}/grads/lora.{name}.B.npy", B.grad)
print("loss", loss.item())

params = [t for pair in lora.values() for t in pair]
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
losses = []
for step in range(args.steps):
    opt.zero_grad()
    l, _ = loss_fn(ids, pos, pad, tokens, dpos, targets, weights)
    l.backward()
    opt.step()
    losses.append(l.item())
    print("step", step, "loss", l.item())
save(f"{args.out}/trajectory/loss.npy", torch.tensor(losses, dtype=DT))
for name, (A, B) in lora.items():
    save(f"{args.out}/trajectory/lora.{name}.A.npy", A)
    save(f"{args.out}/trajectory/lora.{name}.B.npy", B)
