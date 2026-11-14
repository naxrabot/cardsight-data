"""Experiment: fingerprint every catalog picture with DINOv2 (a much stronger picture model than the
phone's current one) so its accuracy can be measured on real livestream frames before the app changes."""
import concurrent.futures as cf, io, os, sys, time, urllib.request
import numpy as np, torch
from PIL import Image

OUT = "out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "cardsight-data (github.com/naxrabot/cardsight-data)"}
S = 224
torch.set_num_threads(os.cpu_count() or 4)

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):                      # x: N×3×224×224, RGB 0..1
        f = self.m.forward_features((x - self.mean) / self.std)
        return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)   # N×768

name = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14"
net = Wrap(torch.hub.load("facebookresearch/dinov2", name)).eval()
with torch.no_grad():
    torch.onnx.export(net, torch.zeros(1, 3, S, S), f"{OUT}/{name}.onnx", input_names=["x"], output_names=["e"],
                      dynamic_axes={"x": {0: "n"}, "e": {0: "n"}}, opset_version=17)
print("onnx", os.path.getsize(f"{OUT}/{name}.onnx") >> 20, "MB", flush=True)

urls = [str(u) for u in np.load("cache/embcache.npz", allow_pickle=False)["urls"]]
if os.environ.get("LIMIT"): urls = urls[: int(os.environ["LIMIT"])]

def fetch(u):
    for a in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(u + "/low.png", headers=UA), timeout=30) as r:
                p = Image.open(io.BytesIO(r.read())).convert("RGB")
            w, h = p.size
            art = p.crop((int(w * .07), int(h * .10), int(w * .93), int(h * .53)))
            t = lambda im: np.asarray(im.resize((S, S), Image.BICUBIC), dtype=np.uint8)
            return t(p), t(art)
        except Exception:
            time.sleep(min(2 ** a, 20))
    return None

full = np.zeros((len(urls), 768), np.float16); art = np.zeros((len(urls), 768), np.float16)
ok = np.zeros(len(urls), bool)
def run(batch):
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad(): return net(x).numpy()
t0 = time.time(); B = 48
with cf.ThreadPoolExecutor(16) as ex:
    buf = []; idx = []
    def flush():
        if not buf: return
        e = run([b[0] for b in buf]); a = run([b[1] for b in buf])
        full[idx] = e; art[idx] = a; ok[idx] = True
        buf.clear(); idx.clear()
    for i, r in enumerate(ex.map(fetch, urls)):
        if r is not None: buf.append(r); idx.append(i)
        if len(buf) >= B: flush()
        if i % 2000 == 0: print(i, len(urls), int(time.time() - t0), "s", flush=True)
    flush()
print("done", int(ok.sum()), "of", len(urls), int(time.time() - t0), "s", flush=True)
np.savez_compressed(f"{OUT}/{name}.npz", urls=np.array(urls), ok=ok, full=full, art=art)
