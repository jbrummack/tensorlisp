"""numpy reference of the VoiceChat audio codec's decoder (RVQVAEModel.decode in NeMo's ear_tts_vae_codec.py, read,
not imported): codes [T, 31] -> prvq.decode (sum of the codebook rows) -> Latent2Wav (ConvTranspose1d with kernel =
stride, ConvNeXt1d blocks with causal depthwise convs, a 1x1 conv to 18 channels = 9 magnitudes + 9 phases) ->
ISTFT (n_fft 16, hop 4, periodic Hann) -> 22.05 kHz waveform, 1764 samples per code frame.

  python3 -I reference_codec.py fetch DIR          reads the decoder and codebooks from the checkpoint into DIR/codec.npz
  python3 -I reference_codec.py decode DIR T [seed]  DIR/codes.npy [T, 31] (random) and DIR/wav.npy
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RATES = (9, 7, 7)          # upsampling of the three stages (the encoder's rates reversed)
CHANNELS = (1536, 768, 384)
KERNEL = 7
BLOCKS = 3
N_FFT, HOP = 16, 4
PREFIX = "tts_model.audio_codec."


def fetch(d):
    from reference_audio import fetch_tensors
    names = [PREFIX + "prvq.mus_list.%d" % i for i in range(31)]
    layers = {0: "convT", 4: "convT", 8: "convT", 12: "conv"}
    for i in range(13):
        if i in layers:
            names.append(PREFIX + "decoder.layers.%d.weight" % i)
        else:
            for n in ("dwconv", "norm", "pwconv1", "pwconv2"):
                names += [PREFIX + "decoder.layers.%d.%s.weight" % (i, n), PREFIX + "decoder.layers.%d.%s.bias" % (i, n)]
    t = fetch_tensors(names)
    np.savez(d + "/codec.npz", **{k[len(PREFIX):].replace(".", "/"): v for k, v in t.items()})


def load(d):
    z = np.load(d + "/codec.npz")
    return {k.replace("/", "."): z[k].astype(np.float64) for k in z.files}


def gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def layer_norm_channels(x, w, b, eps=1e-6):       # x [C, T]
    m = x.mean(0, keepdims=True)
    v = ((x - m) ** 2).mean(0, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w[:, None] + b[:, None]


def convnext(x, w, i, tap=lambda *a: None):         # x [C, T]
    p = "decoder.layers.%d." % i
    c, t = x.shape
    pad = np.concatenate([np.zeros((c, KERNEL - 1)), x], 1)
    k = w[p + "dwconv.weight"][:, 0, :]             # [C, 7]
    y = sum(k[:, j:j + 1] * pad[:, j:j + t] for j in range(KERNEL)) + w[p + "dwconv.bias"][:, None]
    tap("cn%d.dw" % i, y.T)
    y = layer_norm_channels(y, w[p + "norm.weight"], w[p + "norm.bias"])
    tap("cn%d.norm" % i, y.T)
    y = w[p + "pwconv1.weight"][:, :, 0] @ y + w[p + "pwconv1.bias"][:, None]
    tap("cn%d.pw1" % i, y.T)
    y = gelu(y)
    tap("cn%d.gelu" % i, y.T)
    y = w[p + "pwconv2.weight"][:, :, 0] @ y + w[p + "pwconv2.bias"][:, None]
    return x + y


def convt(x, weight, s):                            # x [Cin, T], weight [Cin, Cout, s] -> [Cout, T*s]
    y = np.einsum("it,ioj->otj", x, weight)
    return y.reshape(weight.shape[1], -1)


def istft(mag_ph, constrain=True):                  # [18, T]
    mag, ph = mag_ph[:9], mag_ph[9:]
    mag = 100.0 * np.exp(-np.logaddexp(0, -mag + math.log(100.0)))
    re = mag * np.cos(ph)
    im = mag * np.sin(ph)
    im[0] = 0
    im[8] = 0
    spec = re + 1j * im
    frames = np.fft.irfft(spec, n=N_FFT, axis=0)    # [16, T]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)     # periodic Hann
    if constrain:
        frames = np.where(frames >= 0, np.minimum(frames, window[:, None]), np.maximum(frames, -window[:, None]))
    frames = frames * window[:, None]
    t = frames.shape[1]
    out = np.zeros((t - 1) * HOP + N_FFT)
    env = np.zeros_like(out)
    for i in range(t):
        out[i * HOP:i * HOP + N_FFT] += frames[:, i]
        env[i * HOP:i * HOP + N_FFT] += window ** 2
    pad = (N_FFT - HOP) // 2
    return out[pad:-pad] / env[pad:-pad]


def decode(w, codes, taps=None):                    # codes [T, 31] -> wav
    tap = (lambda name, v: taps.__setitem__(name, v.astype(np.float32))) if taps is not None else (lambda *a: None)
    z = sum(w["prvq.mus_list.%d" % i][codes[:, i]] for i in range(31))     # [T, 512]
    tap("codec.z", z)
    x = z.T
    layer = 0
    for stage in range(3):
        x = convt(x, w["decoder.layers.%d.weight" % layer], RATES[stage])
        tap("codec.up%d" % stage, x.T)
        layer += 1
        for _ in range(BLOCKS):
            x = convnext(x, w, layer, tap)
            layer += 1
        tap("codec.stage%d" % stage, x.T)
    x = w["decoder.layers.12.weight"][:, :, 0] @ x
    tap("codec.spec", x.T)
    return istft(x)


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "fetch":
        fetch(d)
    else:
        t = int(sys.argv[3])
        codes = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0).integers(0, 1024, (t, 31))
        np.save(d + "/codes.npy", codes.astype(np.float32))
        np.save(d + "/wav.npy", decode(load(d), codes).astype(np.float32))
