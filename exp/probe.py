"""Fingerprint a list of pictures (exp/probe.txt) with each candidate model, for offline comparison."""
import sys, os, numpy as np, urllib.request, concurrent.futures as cf
sys.path.insert(0, ".")
import onnxruntime as ort
from build_data import http, img_url, net_views
urls = [u.strip() for u in open("exp/probe.txt") if u.strip()]
os.makedirs("out", exist_ok=True)
def fetch(u):
    try: return u, net_views(http(img_url(u), tries=4, timeout=30))
    except Exception: return u, None
with cf.ThreadPoolExecutor(12) as ex: got = [(u, v) for u, v in ex.map(fetch, urls) if v is not None]
print(len(got), "of", len(urls), flush=True)
x = np.concatenate([v for _, v in got]).transpose(0, 3, 1, 2).astype(np.float32)
for tag in sys.argv[1:]:
    urllib.request.urlretrieve(f"https://github.com/naxrabot/cardsight-data/releases/download/exp/{tag}_cardnet.onnx", f"/tmp/{tag}.onnx")
    s = ort.InferenceSession(f"/tmp/{tag}.onnx", providers=["CPUExecutionProvider"])
    e = np.concatenate([s.run(None, {"x": x[i:i + 64]})[0] for i in range(0, len(x), 64)]).reshape(len(got), 2, -1).sum(1)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.savez_compressed(f"out/{tag}_probe.npz", urls=np.array([u for u, _ in got]), vecs=e.astype(np.float16))
