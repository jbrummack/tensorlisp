"""numpy references for the VoiceChat audio front end, from NeMo's code (read, not imported: there is no torch here).

  python3 -I reference_audio.py fetch DIR      reads the featurizer's fb [1,128,257] and window [400] from the checkpoint
  python3 -I reference_audio.py mel DIR WAV    DIR/mel.npy [frames, 128]: FilterbankFeatures.forward in eval mode
"""
import json, os, struct, sys, urllib.request, wave
import numpy as np

URL = "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/resolve/main/model.safetensors"


def fetch_range(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b - 1}", "Authorization": "Bearer " + os.environ["HF_TOKEN"]})
    return urllib.request.urlopen(req).read()


def fetch_tensors(names):
    n = struct.unpack("<Q", fetch_range(0, 8))[0]
    header = json.loads(fetch_range(8, 8 + n))
    out = {}
    for name in names:
        t = header[name]
        a, b = t["data_offsets"]
        dt = {"F32": "<f4"}[t["dtype"]]
        out[name] = np.frombuffer(fetch_range(8 + n + a, 8 + n + b), dt).reshape(t["shape"])
    return out


def read_wav(path):
    w = wave.open(path)
    assert w.getsampwidth() == 2 and w.getframerate() == 16000 and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def mel(x, fb, window, preemph=0.97, n_fft=512, hop=160, guard=2.0 ** -24):
    x = x.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - preemph * x[:-1]])
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    win = np.zeros(n_fft, np.float32)
    off = (n_fft - len(window)) // 2
    win[off:off + len(window)] = window
    n = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)])
    power = np.abs(np.fft.rfft(frames.astype(np.float64), axis=1)) ** 2          # [n, 257]
    return np.log(power @ fb[0].T.astype(np.float64) + guard).astype(np.float32)  # [n, 128]


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "fetch":
        t = fetch_tensors(["stt_model.perception.preprocessor.featurizer.fb", "stt_model.perception.preprocessor.featurizer.window"])
        np.save(f"{d}/fb.npy", t["stt_model.perception.preprocessor.featurizer.fb"])
        np.save(f"{d}/window.npy", t["stt_model.perception.preprocessor.featurizer.window"])
    elif cmd == "mel":
        np.save(f"{d}/mel.npy", mel(read_wav(sys.argv[3]), np.load(f"{d}/fb.npy"), np.load(f"{d}/window.npy")))
