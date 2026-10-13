"""A tiny random Nemotron-H with the real tensor names, and a plain numpy implementation of
modeling_nemotron_h.py's math (float64, no cache: the whole sequence every time) to check
backbone.ss against, since there is no quantized (or runnable) reference of the real model here.

  python3 -I reference_tiny.py make DIR       DIR/tiny.safetensors, DIR/{token,pos,logits}.npy,
                                              DIR/backbone_tiny.ss (backbone.ss with the tiny dimensions)
  python3 -I reference_tiny.py greedy DIR IDS...   prints the greedy continuation of IDS

Dimensions: hidden 64, Mamba2 with 8 heads of 4 (d_inner 32), 8 groups, state 2, conv 4;
attention 2 query heads + 1 KV head of 128; MLP 96; layers "M-M*-"; the real vocab (131072)
so the real tokenizer's ids work.
"""
import json, struct, sys
import numpy as np

H, VOCAB, NH, MHD, G, S, DCONV, HD, QH, KVH, INTER = 64, 131072, 8, 4, 8, 2, 4, 128, 2, 1, 96
PATTERN = "M-M*-"
EPS = 1e-5
DI = NH * MHD
CD = DI + 2 * G * S


def make_weights(seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s, std=0.08: (r.standard_normal(s) * std).astype(np.float32)
    w = {"embed_tokens.weight": n(VOCAB, H, std=0.5), "lm_head.weight": n(VOCAB, H, std=0.3),
         "llm.norm_f.weight": 1 + n(H, std=0.1)}
    for i, k in enumerate(PATTERN):
        p = f"llm.layers.{i}."
        w[p + "norm.weight"] = 1 + n(H, std=0.1)
        m = p + "mixer."
        if k == "M":
            w[m + "in_proj.weight"] = n(DI + CD + NH, H)
            w[m + "conv1d.weight"] = n(CD, 1, DCONV, std=0.3)
            w[m + "conv1d.bias"] = n(CD, std=0.1)
            w[m + "A_log"] = np.log(np.arange(1, NH + 1)).astype(np.float32)
            w[m + "D"] = 1 + n(NH, std=0.1)
            w[m + "dt_bias"] = n(NH, std=0.5)
            w[m + "norm.weight"] = 1 + n(DI, std=0.1)
            w[m + "out_proj.weight"] = n(H, DI)
        elif k == "*":
            w[m + "q_proj.weight"] = n(QH * HD, H)
            w[m + "k_proj.weight"] = n(KVH * HD, H)
            w[m + "v_proj.weight"] = n(KVH * HD, H)
            w[m + "o_proj.weight"] = n(H, QH * HD)
        else:
            w[m + "up_proj.weight"] = n(INTER, H)
            w[m + "down_proj.weight"] = n(H, INTER)
    w.update(make_encoder_weights(r))
    return w


# --- the speech encoder (audio.ss): tiny dimensions, same names as the converted checkpoint
E_LAYERS, E_D, E_HEADS, E_FF, E_LEFT, E_CH, E_K, MELS = 2, 64, 2, 128, 5, 8, 9, 128
E_HD = E_D // E_HEADS


def make_encoder_weights(r):
    n = lambda *s, std=0.08: (r.standard_normal(s) * std).astype(np.float32)
    w = {}
    pe = "enc.pre_encode."
    w[pe + "conv.0.weight"] = n(E_CH, 1, 3, 3, std=0.3); w[pe + "conv.0.bias"] = n(E_CH, std=0.1)
    for i in (2, 5):
        w[pe + f"conv.{i}.weight"] = n(E_CH, 1, 3, 3, std=0.3); w[pe + f"conv.{i}.bias"] = n(E_CH, std=0.1)
    for i in (3, 6):
        w[pe + f"conv.{i}.weight"] = n(E_CH, E_CH, 1, 1, std=0.3); w[pe + f"conv.{i}.bias"] = n(E_CH, std=0.1)
    w[pe + "out.weight"] = n(E_D, 17 * E_CH, std=0.1); w[pe + "out.bias"] = n(E_D, std=0.1)
    for l in range(E_LAYERS):
        p = f"enc.layers.{l}."
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            w[p + name + ".weight"] = 1 + n(E_D, std=0.1); w[p + name + ".bias"] = n(E_D, std=0.1)
        for ff in ("feed_forward1", "feed_forward2"):
            w[p + ff + ".linear1.weight"] = n(E_FF, E_D); w[p + ff + ".linear2.weight"] = n(E_D, E_FF)
        for a in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            w[p + "self_attn." + a + ".weight"] = n(E_D, E_D, std=0.15)
        w[p + "self_attn.pos_bias_u"] = n(E_HEADS, E_HD, std=0.3); w[p + "self_attn.pos_bias_v"] = n(E_HEADS, E_HD, std=0.3)
        w[p + "conv.pointwise_conv1.weight"] = n(2 * E_D, E_D, 1, std=0.15)
        w[p + "conv.depthwise_conv.weight"] = n(E_D, 1, E_K, std=0.3)
        w[p + "conv.batch_norm.weight"] = 1 + n(E_D, std=0.1); w[p + "conv.batch_norm.bias"] = n(E_D, std=0.1)
        w[p + "conv.pointwise_conv2.weight"] = n(E_D, E_D, 1, std=0.15)
    w["perc.proj.weight"] = n(H, E_D, std=0.1); w["perc.proj.bias"] = n(H, std=0.1)
    return w


def ln(x, w, b, eps=EPS):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def conv2d(x, w, b, groups=1):
    """x [C, T, F] padded 2 before / 1 after on both axes, kernel 3, stride 2 (NeMo's CausalConv2D), w [Co, Ci/g, kh, kw]."""
    x = np.pad(x, ((0, 0), (2, 1), (2, 1)))
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))[:, ::2, ::2]   # [C, T', F', 3, 3]
    co = w.shape[0]; cg = x.shape[0] // groups; og = co // groups
    out = np.concatenate([np.einsum("cthij,oc ij->oth", win[g * cg:(g + 1) * cg], w[g * og:(g + 1) * og]) for g in range(groups)])
    return out + b[:, None, None]


def encoder_forward(w, mel):
    w = {k: v.astype(np.float64) for k, v in w.items()}
    pe = "enc.pre_encode."
    x = mel.astype(np.float64)[None]                                                    # [1, T, 128]
    x = np.maximum(conv2d(x, w[pe + "conv.0.weight"], w[pe + "conv.0.bias"]), 0)
    for dw, pw in ((2, 3), (5, 6)):
        x = conv2d(x, w[pe + f"conv.{dw}.weight"], w[pe + f"conv.{dw}.bias"], groups=E_CH)
        x = np.maximum(np.einsum("oc,cth->oth", w[pe + f"conv.{pw}.weight"][:, :, 0, 0], x) + w[pe + f"conv.{pw}.bias"][:, None, None], 0)
    c, t, f = x.shape
    x = x.transpose(1, 0, 2).reshape(t, c * f) @ w[pe + "out.weight"].T + w[pe + "out.bias"]
    # relative positions: t-1 down to -(t-1), sin at even channels, cos at odd
    pos = np.arange(t - 1, -t, -1, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, E_D, 2) * -(np.log(10000.0) / E_D))
    pemb = np.zeros((2 * t - 1, E_D)); pemb[:, 0::2] = np.sin(pos * div); pemb[:, 1::2] = np.cos(pos * div)
    i, j = np.arange(t)[:, None], np.arange(t)[None]
    allowed = (i - j >= 0) & (i - j <= E_LEFT)
    silu = lambda v: v / (1 + np.exp(-v))
    for l in range(E_LAYERS):
        p = f"enc.layers.{l}."
        g = lambda name: w[p + name]
        ff = lambda v, n: silu(v @ g(n + ".linear1.weight").T) @ g(n + ".linear2.weight").T
        x = x + 0.5 * ff(ln(x, g("norm_feed_forward1.weight"), g("norm_feed_forward1.bias")), "feed_forward1")
        a = ln(x, g("norm_self_att.weight"), g("norm_self_att.bias"))
        q = (a @ g("self_attn.linear_q.weight").T).reshape(t, E_HEADS, E_HD)
        k = (a @ g("self_attn.linear_k.weight").T).reshape(t, E_HEADS, E_HD)
        v = (a @ g("self_attn.linear_v.weight").T).reshape(t, E_HEADS, E_HD)
        pp = (pemb @ g("self_attn.linear_pos.weight").T).reshape(-1, E_HEADS, E_HD)
        out = np.zeros((t, E_HEADS, E_HD))
        for h in range(E_HEADS):
            qu = q[:, h] + g("self_attn.pos_bias_u")[h]; qv = q[:, h] + g("self_attn.pos_bias_v")[h]
            ac = qu @ k[:, h].T
            bd = qv @ pp[:, h].T                                                          # [t, 2t-1]
            # NeMo's rel_shift: pad one column on the left, reinterpret, drop the first row
            b = np.pad(bd, ((0, 0), (1, 0))).reshape(2 * t, t)[1:].reshape(t, 2 * t - 1)[:, :t]
            s = (ac + b) / np.sqrt(E_HD)
            s = np.where(allowed, s, -np.inf)
            pr = np.exp(s - s.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            out[:, h] = pr @ v[:, h]
        x = x + out.reshape(t, E_D) @ g("self_attn.linear_out.weight").T
        c_ = ln(x, g("norm_conv.weight"), g("norm_conv.bias"))
        h1 = c_ @ g("conv.pointwise_conv1.weight")[:, :, 0].T                             # [t, 2d]
        h1 = h1[:, :E_D] / (1 + np.exp(-h1[:, E_D:])) if False else h1[:, :E_D] * (1 / (1 + np.exp(-h1[:, E_D:])))
        pad = np.concatenate([np.zeros((E_K - 1, E_D)), h1])
        kw = g("conv.depthwise_conv.weight")[:, 0, :]                                     # [d, K]
        dwc = np.stack([sum(kw[:, m] * pad[s_ + m] for m in range(E_K)) for s_ in range(t)])
        h2 = silu(ln(dwc, g("conv.batch_norm.weight"), g("conv.batch_norm.bias"))) @ g("conv.pointwise_conv2.weight")[:, :, 0].T
        x = x + h2
        x = x + 0.5 * ff(ln(x, g("norm_feed_forward2.weight"), g("norm_feed_forward2.bias")), "feed_forward2")
        x = ln(x, g("norm_out.weight"), g("norm_out.bias"))
    return x, x @ w["perc.proj.weight"].T + w["perc.proj.bias"]


def write_safetensors(path, tensors):
    header, blob, off = {}, b"", 0
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype="<f4").tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    open(path, "wb").write(struct.pack("<Q", len(hb)) + hb + blob)


def rms(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def mamba(x, w, m):
    T = len(x)
    proj = x @ w[m + "in_proj.weight"].T
    z, xbc, dt = proj[:, :DI], proj[:, DI:DI + CD], proj[:, DI + CD:]
    k = w[m + "conv1d.weight"][:, 0, :]
    pad = np.concatenate([np.zeros((DCONV - 1, CD)), xbc])
    conv = np.stack([sum(k[:, j] * pad[t + j] for j in range(DCONV)) for t in range(T)]) + w[m + "conv1d.bias"]
    xbc = silu(conv)
    xs = xbc[:, :DI].reshape(T, NH, MHD)
    B = xbc[:, DI:DI + G * S].reshape(T, G, S)
    C = xbc[:, DI + G * S:].reshape(T, G, S)
    dt = softplus(dt + w[m + "dt_bias"])
    A = -np.exp(w[m + "A_log"].astype(np.float64))
    h = np.zeros((NH, MHD, S))
    g = np.arange(NH) // (NH // G)
    y = np.zeros((T, NH, MHD))
    for t in range(T):
        dA = np.exp(dt[t] * A)
        h = h * dA[:, None, None] + (dt[t][:, None] * xs[t])[:, :, None] * B[t][g][:, None, :]
        y[t] = (h * C[t][g][:, None, :]).sum(-1)
    y = y + w[m + "D"][None, :, None] * xs
    gated = y.reshape(T, DI) * silu(z)
    gs = DI // G
    normed = rms(gated.reshape(T, G, gs), w[m + "norm.weight"].reshape(G, gs)).reshape(T, DI)
    return normed @ w[m + "out_proj.weight"].T


def attention(x, w, m):
    T = len(x)
    q = (x @ w[m + "q_proj.weight"].T).reshape(T, QH, HD)
    k = (x @ w[m + "k_proj.weight"].T).reshape(T, KVH, HD)
    v = (x @ w[m + "v_proj.weight"].T).reshape(T, KVH, HD)
    out = np.zeros((T, QH, HD))
    for h in range(QH):
        kv = h // (QH // KVH)
        s = q[:, h] @ k[:, kv].T / np.sqrt(HD)
        s = np.where(np.tri(T, dtype=bool), s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, kv]
    return out.reshape(T, QH * HD) @ w[m + "o_proj.weight"].T


def forward(w, ids):
    x = w["embed_tokens.weight"][ids].astype(np.float64)
    w = {k: v.astype(np.float64) for k, v in w.items()}
    for i, k in enumerate(PATTERN):
        h = rms(x, w[f"llm.layers.{i}.norm.weight"])
        m = f"llm.layers.{i}.mixer."
        if k == "M":
            x = x + mamba(h, w, m)
        elif k == "*":
            x = x + attention(h, w, m)
        else:
            x = x + (np.maximum(h @ w[m + "up_proj.weight"].T, 0) ** 2) @ w[m + "down_proj.weight"].T
    return rms(x[-1], w["llm.norm_f.weight"]) @ w["lm_head.weight"].T


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    w = make_weights()
    if cmd == "make":
        src = open(__file__.rsplit("/", 1)[0] + "/backbone.ss").read()
        a, b = src.index(";; CONFIG-BEGIN"), src.index(";; CONFIG-END")
        tiny = f"""(define pattern "{PATTERN}")
(define hidden {H})
(define n-heads {NH})
(define mamba-head-dim {MHD})
(define n-group {G})
(define d-state {S})
(define q-heads {QH})
(define kv-heads {KVH})
(define head-dim {HD})
"""
        open(f"{out}/backbone_tiny.ss", "w").write(src[:a] + ";; CONFIG-BEGIN\n" + tiny + src[b:])
        write_safetensors(f"{out}/tiny.safetensors", w)
        ids = np.array([5, 1234, 77, 98765, 4, 31337, 2024, 9])
        np.save(f"{out}/token.npy", ids[None].astype(np.float32))
        np.save(f"{out}/pos.npy", np.arange(len(ids))[None].astype(np.float32))
        np.save(f"{out}/logits.npy", forward(w, ids)[None].astype(np.float32))
        mel = (np.random.default_rng(1).standard_normal((150, MELS)) * 2 - 8).astype(np.float32)
        emb, proj = encoder_forward(w, mel)
        np.save(f"{out}/mel.npy", mel); np.save(f"{out}/emb.npy", emb.astype(np.float32)); np.save(f"{out}/proj.npy", proj.astype(np.float32))
        src = open(__file__.rsplit("/", 1)[0] + "/audio.ss").read()
        a, b = src.index(";; ENC-CONFIG-BEGIN"), src.index(";; ENC-CONFIG-END")
        cfg = f"""(define enc-layers {E_LAYERS})
(define enc-d {E_D})
(define enc-heads {E_HEADS})
(define enc-left {E_LEFT})
(define sub-channels {E_CH})
"""
        open(f"{out}/audio_tiny.ss", "w").write(src[:a] + ";; ENC-CONFIG-BEGIN\n" + cfg + src[b:])
    else:
        ids = [int(a) for a in sys.argv[3:]]
        for _ in range(int(__import__("os").environ.get("N", 12))):
            ids.append(int(np.argmax(forward(w, np.array(ids)))))
        print(ids)
