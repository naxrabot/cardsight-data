"""Trains CardSight's own picture model: given a livestream view of a card (small, tilted, glare, marker,
fingers, chat over it, other cards behind) it must land next to that card's clean catalog picture.
Every catalog picture is its own class; views are made by aug.stream_view."""
import concurrent.futures as cf, gzip, io, json, math, os, sys, time, urllib.request
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aug
import timm  # imported up front so a broken install fails in seconds, not after the downloads

BACKBONE = os.environ.get("BACKBONE", "mobilenetv4_conv_medium.e500_r256_in1k")
BUDGET = float(os.environ.get("BUDGET_MIN", "200")) * 60          # training time
LIMIT = int(os.environ.get("LIMIT", "0"))
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"
DIM, BATCH, WORKERS = 256, int(os.environ.get("BATCH", "96")), int(os.environ.get("WORKERS", "3"))
OUT, CACHE = "out", "imgcache"
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "cardsight-data (github.com/naxrabot/cardsight-data)"}
T0 = time.time()
LINES = []
def log(*a):
    line = f"[{int(time.time() - T0):5d}s] " + " ".join(str(x) for x in a)
    LINES.append(line); print(line, flush=True)

TAG = os.environ.get("TAG", "x")
_last_up = [0.0]
def publish(names, every=0):
    """Upload progress / checkpoints to the 'exp' release so a long run can be watched and isn't lost."""
    if not os.environ.get("GH_TOKEN") or time.time() - _last_up[0] < every: return
    _last_up[0] = time.time()
    import shutil, subprocess
    try:
        open(f"{OUT}/progress.txt", "w").write("\n".join(LINES))
        paths = []
        for n in ["progress.txt"] + names:
            d = f"/tmp/{TAG}_{n}"; shutil.copyfile(f"{OUT}/{n}", d); paths.append(d)
        subprocess.run(["gh", "release", "upload", "exp", *paths, "--clobber"], timeout=600, check=False, capture_output=True)
    except Exception as e:
        print("publish failed:", e, flush=True)

# ---------------------------------------------------------------- pictures
def load_images(urls):
    """JPEG bytes per url (kept between runs in imgcache/)."""
    store = {}
    p = os.path.join(CACHE, "imgs.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False); blob = z["blob"].tobytes(); off = z["off"]
        for i, u in enumerate(z["urls"]): store[str(u)] = blob[off[i]:off[i + 1]]
        log("picture cache:", len(store))
    todo = [u for u in urls if u not in store]
    def fetch(u):
        for a in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(u + "/low.png", headers=UA), timeout=30) as r:
                    im = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
                im = cv2.resize(im, (245, 337), interpolation=cv2.INTER_AREA)
                return u, cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
            except Exception:
                time.sleep(min(2 ** a, 15))
        return u, None
    if todo:
        with cf.ThreadPoolExecutor(24) as ex:
            for n, (u, b) in enumerate(ex.map(fetch, todo), 1):
                if b: store[u] = b
                if n % 4000 == 0: log(f"pictures {n}/{len(todo)}")
        us = sorted(store); off = np.cumsum([0] + [len(store[u]) for u in us])
        np.savez(p, urls=np.array(us), off=off, blob=np.frombuffer(b"".join(store[u] for u in us), np.uint8))
    return store

def decode(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

class Views(torch.utils.data.Dataset):
    def __init__(self, blobs, seed=0, fixed=False): self.blobs, self.seed, self.fixed, self.pool, self.rng = blobs, seed, fixed, [], None
    def __len__(self): return len(self.blobs)
    def __getitem__(self, i):
        if self.rng is None:
            w = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, w.id if w else 0, 0 if self.fixed else int(time.time())])
        rng = np.random.default_rng([self.seed, i]) if self.fixed else self.rng
        card = decode(self.blobs[i])
        if len(self.pool) < 48: self.pool.append(card)
        else: self.pool[int(self.rng.integers(0, 48))] = card
        v = aug.stream_view(card, rng, self.pool)
        return torch.from_numpy(v).permute(2, 0, 1), i

class Clean(torch.utils.data.Dataset):
    def __init__(self, blobs): self.blobs = blobs
    def __len__(self): return len(self.blobs)
    def __getitem__(self, i):
        c = decode(self.blobs[i]); a = aug.clean_view(c)
        # a second, "livestream quality" copy: small and soft
        s = cv2.resize(cv2.resize(c, (82, 112), interpolation=cv2.INTER_AREA), (aug.W, aug.H), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.stack([a, s])).permute(0, 3, 1, 2), i

# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = timm.create_model(BACKBONE, pretrained=PRETRAINED, num_classes=0)
        with torch.no_grad(): feat = self.body.eval()(torch.zeros(1, 3, aug.H, aug.W)).shape[1]
        self.fc = nn.Linear(feat, DIM); self.bn = nn.BatchNorm1d(DIM)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
    def forward(self, x):                      # x: N×3×256×192, RGB 0..255 (float)
        return F.normalize(self.bn(self.fc(self.body((x - self.mean) / self.std))), dim=1)

@torch.no_grad()
def embed_gallery(net, blobs):
    net.eval(); out = np.zeros((len(blobs), DIM), np.float32)
    dl = torch.utils.data.DataLoader(Clean(blobs), batch_size=128, num_workers=WORKERS)
    for x, i in dl:
        n = x.shape[0]; e = net(x.view(-1, 3, aug.H, aug.W).float().contiguous(memory_format=torch.channels_last)).view(n, 2, DIM)
        out[i.numpy()] = F.normalize(e.sum(1), dim=1).numpy()
    return out

@torch.no_grad()
def embed_views(net, x):
    net.eval(); return net(x.float().contiguous(memory_format=torch.channels_last)).numpy()

def main():
    torch.manual_seed(0)
    urls = [str(u) for u in np.load("cache/embcache.npz", allow_pickle=False)["urls"]]
    if LIMIT: urls = urls[:LIMIT]
    store = load_images(urls)
    urls = [u for u in urls if u in store]; blobs = [store[u] for u in urls]; N = len(urls)
    if os.environ.get("FETCH_ONLY"): log("pictures ready:", N); return
    log("pictures:", N, "backbone:", BACKBONE)
    publish([])
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    net = Net().to(memory_format=torch.channels_last)
    cls = nn.Parameter(torch.randn(N, DIM) * 0.01)
    opt = torch.optim.AdamW([{"params": net.body.parameters(), "lr": 4e-4}, {"params": list(net.fc.parameters()) + list(net.bn.parameters()) + [cls], "lr": 2e-3}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    dl = torch.utils.data.DataLoader(Views(blobs, seed=1), batch_size=BATCH, shuffle=True, num_workers=WORKERS, drop_last=True, persistent_workers=True)
    S, M = 30.0, 0.20
    t_start = time.time(); step = 0; ep = 0; run_loss = run_acc = 0.0
    while time.time() - t_start < BUDGET:
        net.train()
        for x, y in dl:
            frac = (time.time() - t_start) / BUDGET
            if frac >= 1: break
            k = min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
            for g, b in zip(opt.param_groups, base): g["lr"] = b * max(k, 0.01)
            e = net(x.float().contiguous(memory_format=torch.channels_last))
            logits = e @ F.normalize(cls, dim=1).t()
            loss = F.cross_entropy(S * (logits - M * F.one_hot(y, N)), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_loss += loss.item(); run_acc += (logits.argmax(1) == y).float().mean().item(); step += 1
            if step % 100 == 0:
                log(f"ep {ep} step {step} loss {run_loss / 100:.3f} acc {run_acc / 100:.3f} lr {k:.3f} {step * BATCH / (time.time() - t_start):.1f} img/s")
                run_loss = run_acc = 0.0
                publish([], every=900)
        ep += 1
        torch.save({"net": net.state_dict()}, f"{OUT}/model.pt")
        publish(["model.pt"], every=2400)
    log("training done:", ep, "epochs", step, "steps")

    # ------------------------------------------------------------ test
    G = embed_gallery(net, blobs); log("gallery embedded")
    report = {"backbone": BACKBONE, "epochs": ep, "steps": step, "pictures": N}
    vn = min(2000, N); idx = np.random.default_rng(7).choice(N, vn, replace=False)
    vds = Views([blobs[i] for i in idx], seed=99, fixed=True)
    q = np.concatenate([embed_views(net, x) for x, _ in torch.utils.data.DataLoader(vds, batch_size=128, num_workers=WORKERS)])
    sims = q @ G.T; rank = (sims > sims[np.arange(vn), idx][:, None]).sum(1)
    report["sim_top1"] = float((rank == 0).mean()); report["sim_top5"] = float((rank < 5).mean()); report["sim_top20"] = float((rank < 20).mean())
    log("simulated livestream views: top1 %.3f top5 %.3f top20 %.3f" % (report["sim_top1"], report["sim_top5"], report["sim_top20"]))
    # real frames
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real")
    truth = json.load(open(os.path.join(real, "truth.json")))
    cat = json.load(gzip.open("cache/catalog.json.gz")); cat = cat["cards"] if isinstance(cat, dict) else cat
    img_of = {(c["i"], c.get("l", "en")): c.get("img") for c in cat}
    row = {u: i for i, u in enumerate(urls)}; report["real"] = {}
    for f, ids in sorted(truth.items()):
        rows = [row[img_of[tuple(t.split("@"))]] for t in ids if img_of.get(tuple(t.split("@"))) in row]
        im = cv2.cvtColor(cv2.imread(os.path.join(real, f)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(cv2.resize(im, (aug.W, aug.H), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None]
        s = embed_views(net, x)[0] @ G.T; o = np.argsort(-s)
        r = min(int((s > s[t]).sum()) for t in rows) if rows else -1
        report["real"][f] = {"rank": r, "top_sim": float(s[o[0]]), "true_sim": float(max(s[t] for t in rows)) if rows else None, "top": urls[o[0]]}
        log("real", f, report["real"][f])
    # ------------------------------------------------------------ export
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, aug.H, aug.W), f"{OUT}/cardnet.onnx", input_names=["x"], output_names=["e"],
                      dynamic_axes={"x": {0: "n"}, "e": {0: "n"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    xs = torch.rand(2, 3, aug.H, aug.W) * 255
    d = np.abs(ort.InferenceSession(f"{OUT}/cardnet.onnx", providers=["CPUExecutionProvider"]).run(None, {"x": xs.numpy()})[0] - net(xs).detach().numpy()).max()
    report["onnx_max_diff"] = float(d); report["onnx_mb"] = os.path.getsize(f"{OUT}/cardnet.onnx") / 1e6
    np.savez_compressed(f"{OUT}/gallery.npz", urls=np.array(urls), vecs=G.astype(np.float16))
    json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
    log("report", json.dumps(report))

if __name__ == "__main__":
    main()
