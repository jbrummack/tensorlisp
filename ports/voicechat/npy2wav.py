"""python3 -I npy2wav.py IN.npy OUT.wav [rate]: float samples in [-1, 1] -> 16-bit mono wav (default 22050 Hz)."""
import sys, wave
import numpy as np
a = np.load(sys.argv[1]).astype(np.float64).ravel()
w = wave.open(sys.argv[2], "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sys.argv[3]) if len(sys.argv) > 3 else 22050)
w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
w.close()
print(len(a) / (int(sys.argv[3]) if len(sys.argv) > 3 else 22050), "s")
