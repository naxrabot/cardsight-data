"""Does keeping more fingerprint detail (PCA dims) improve accuracy? Uses the full-size fingerprint cache."""
import io, random, sys, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from build_data import make_embedder, http, art_box
from eval_pictures_lib import stream_view

BASE = "https://github.com/naxrabot/cardsight-data/releases/download/latest/"
open("m.tflite", "wb").write(urllib.request.urlopen(BASE + "embedder.tflite").read())
open("c.npz", "wb").write(urllib.request.urlopen(BASE + "embcache.npz").read())
z = np.load("c.npz"); urls = [str(u) for u in z["urls"]]
XF = z["vecs"].astype(np.float32); XA = z["arts"].astype(np.float32)
embed = make_embedder("m.tflite")
random.seed(5)
qi = random.sample(range(len(urls)), 300)
def fetch(i):
    try: return i, http(urls[i] + "/high.png", tries=3, timeout=30)
    except Exception: return i, None
with cf.ThreadPoolExecutor(12) as ex:
    imgs = [(i, Image.open(io.BytesIO(b)).convert("RGB").resize((367, 512))) for i, b in ex.map(fetch, qi) if b]
views = [(i, stream_view(im, 220)) for i, im in imgs]
QF = np.stack([embed(v) for _, v in views]); QA = np.stack([embed(art_box(v)) for _, v in views])
truth = np.array([i for i, _ in views])
def pca(X, k):
    m = X.mean(0); _, _, vt = np.linalg.svd(X - m, full_matrices=False); return m, vt[:k].T
def run(k):
    if k:
        mF, cF = pca(XF, k); mA, cA = pca(XA, k)
        def P(X, m, c):
            Y = (X - m) @ c; return Y / np.linalg.norm(Y, axis=1, keepdims=True)
        GF, GA, qf, qa = P(XF, mF, cF), P(XA, mA, cA), P(QF, mF, cF), P(QA, mA, cA)
    else:
        GF, GA, qf, qa = XF, XA, QF, QA
    S = (qf @ GF.T + qa @ GA.T) / 2
    o = np.argsort(-S, axis=1)[:, :5]
    return 100 * np.mean(o[:, 0] == truth), 100 * np.mean([t in r for t, r in zip(truth, o)])
for k in (128, 256, 384, 0):
    t1, t5 = run(k)
    print(f"dims {k or 'full'}: top-1 {t1:.0f}%  top-5 {t5:.0f}%", flush=True)
