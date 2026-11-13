"""Accuracy test of the published picture fingerprints (exactly what the phone uses), against the
full catalog, using fake 'livestream' views of random cards."""
import gzip, io, json, random, struct, sys, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
sys.path.insert(0, ".")
from build_data import make_embedder, http, art_box

BASE = "https://github.com/naxrabot/cardsight-data/releases/download/latest/"
def dl(n): return urllib.request.urlopen(BASE + n).read()
open("m.tflite", "wb").write(dl("embedder.tflite"))
cat = json.loads(gzip.decompress(dl("catalog.json.gz")))["cards"]
b = dl("emb.bin"); assert b[:4] == b"CSE2"; off = 4
D, K, N, sF, sA = struct.unpack_from("<iiiff", b, off); off += 20
def arr(n, dt="<f4", size=4):
    global off
    a = np.frombuffer(b, dt, n, off); off += n * size; return a
mF = arr(D); cF = arr(D * K).reshape(D, K); mA = arr(D); cA = arr(D * K).reshape(D, K)
QF = arr(N * K, np.int8, 1).reshape(N, K).astype(np.float32)
QA = arr(N * K, np.int8, 1).reshape(N, K).astype(np.float32)
embed = make_embedder("m.tflite")
rows = {}
for c in cat:
    if c.get("e", -1) >= 0: rows.setdefault(c["e"], c)
random.seed(11)
sample = random.sample(list(rows.items()), 300)

def fetch(item):
    row, c = item
    try: return row, c, http(c["img"] + "/high.png", tries=3, timeout=30)
    except Exception: return row, c, None

def stream_view(card, card_px):
    w, h = card.size
    bg = Image.new("RGB", (int(w * 1.5), int(h * 1.35)), tuple(random.randint(20, 120) for _ in range(3)))
    d = ImageDraw.Draw(bg)
    for _ in range(6):
        x, y = random.randint(0, bg.width), random.randint(0, bg.height)
        d.ellipse([x, y, x + 120, y + 120], fill=tuple(random.randint(60, 200) for _ in range(3)))
    rc = card.rotate(random.uniform(-6, 6), expand=True, fillcolor=(0, 0, 0))
    bg.paste(rc, ((bg.width - rc.width) // 2 + random.randint(-10, 10), (bg.height - rc.height) // 2 + random.randint(-10, 10)))
    g = Image.new("L", bg.size, 0); gd = ImageDraw.Draw(g)
    gx = random.randint(0, bg.width)
    gd.ellipse([gx - 80, -40, gx + 80, bg.height // 2], fill=random.randint(30, 90))
    bg = Image.composite(Image.new("RGB", bg.size, (255, 255, 255)), bg, g.filter(ImageFilter.GaussianBlur(40)))
    bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.7, 1.15))
    s = card_px / w
    small = bg.resize((int(bg.width * s), int(bg.height * s)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.5))
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=40)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    cw, ch = out.width / 1.5, out.height / 1.35
    m = random.uniform(0.0, 0.05)
    l, t = (out.width - cw) / 2 - cw * m, (out.height - ch) / 2 - ch * m
    return out.crop((int(l), int(t), int(l + cw * (1 + 2 * m)), int(t + ch * (1 + 2 * m))))

def proj(v, m, c):
    q = (v - m) @ c; return q / (np.linalg.norm(q) + 1e-9)

with cf.ThreadPoolExecutor(12) as ex:
    data = [(r, c, Image.open(io.BytesIO(x)).convert("RGB").resize((367, 512))) for r, c, x in ex.map(fetch, sample) if x]
print(f"gallery: all {N} catalog pictures; queries: {len(data)}", flush=True)
for px in (300, 200):
    t1 = t5 = 0
    by_lang = {}
    for row, c, im in data:
        q = stream_view(im, px)
        sims = (QF @ proj(embed(q), mF, cF) / sF + QA @ proj(embed(art_box(q)), mA, cA) / sA) / 2
        o = np.argsort(-sims)[:5]
        ok1 = int(o[0] == row); t1 += ok1; t5 += int(row in o)
        l = by_lang.setdefault(c["l"], [0, 0]); l[0] += ok1; l[1] += 1
    n = len(data)
    langs = ", ".join(f"{k} {100*a/b:.0f}% of {b}" for k, (a, b) in by_lang.items())
    print(f"card {px}px wide: exact top-1 {100*t1/n:.0f}%, in top-5 {100*t5/n:.0f}%  ({langs})", flush=True)
