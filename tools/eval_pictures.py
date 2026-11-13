"""Accuracy test: fake 'livestream' views of random cards, then see if the phone's search finds them."""
import gzip, io, json, random, struct, sys, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
sys.path.insert(0, ".")
from build_data import make_embedder, http

BASE = "https://github.com/naxrabot/cardsight-data/releases/download/latest/"
def dl(n): return urllib.request.urlopen(BASE + n).read()
open("m.tflite", "wb").write(dl("embedder.tflite"))
cat = json.loads(gzip.decompress(dl("catalog.json.gz")))["cards"]
b = dl("emb.bin"); off = 4
D, K, N, scale = struct.unpack_from("<iiif", b, off); off += 16
mean = np.frombuffer(b, "<f4", D, off); off += 4 * D
comps = np.frombuffer(b, "<f4", D * K, off).reshape(D, K); off += 4 * D * K
Q = np.frombuffer(b, np.int8, N * K, off).reshape(N, K).astype(np.float32)
embed = make_embedder("m.tflite")
rows = {}
for c in cat:
    if c.get("e", -1) >= 0: rows.setdefault(c["e"], c)
random.seed(7)
sample = random.sample(list(rows.items()), 250)

def stream_view(card, hard):
    w, h = card.size
    bg = Image.new("RGB", (int(w * 1.5), int(h * 1.35)), tuple(random.randint(20, 120) for _ in range(3)))
    # hand/sleeve-ish clutter
    d = ImageDraw.Draw(bg)
    for _ in range(6):
        x, y = random.randint(0, bg.width), random.randint(0, bg.height)
        d.ellipse([x, y, x + 80, y + 80], fill=tuple(random.randint(60, 200) for _ in range(3)))
    ang = random.uniform(-12, 12) if hard else random.uniform(-4, 4)
    rc = card.rotate(ang, expand=True, fillcolor=(0, 0, 0))
    bg.paste(rc, ((bg.width - rc.width) // 2 + random.randint(-15, 15), (bg.height - rc.height) // 2 + random.randint(-15, 15)))
    # glare
    g = Image.new("L", bg.size, 0); gd = ImageDraw.Draw(g)
    gx = random.randint(0, bg.width)
    gd.ellipse([gx - 60, -40, gx + 60, bg.height // 2], fill=random.randint(40, 110) if hard else 30)
    bg = Image.composite(Image.new("RGB", bg.size, (255, 255, 255)), bg, g.filter(ImageFilter.GaussianBlur(30)))
    bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.7, 1.15))
    bg = ImageEnhance.Color(bg).enhance(random.uniform(0.8, 1.2))
    small = bg.resize((bg.width // (3 if hard else 2), bg.height // (3 if hard else 2)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(0.7 if hard else 0.3))
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=30 if hard else 55)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    # crop roughly to the card like the app's card finder would (with a margin)
    cw, ch = out.width / 1.5, out.height / 1.35
    l, t = (out.width - cw) / 2 - cw * 0.06, (out.height - ch) / 2 - ch * 0.06
    return out.crop((int(l), int(t), int(l + cw * 1.12), int(t + ch * 1.12)))

for mode in ("normal", "hard"):
    top1 = top5 = 0; n = 0
    for row, c in sample:
        raw = http(c["img"] + "/low.png", tries=3)
        if not raw: continue
        card = Image.open(io.BytesIO(raw)).convert("RGB")
        v = embed(stream_view(card, mode == "hard"))
        q = (v - mean) @ comps; q /= np.linalg.norm(q) + 1e-9
        sims = Q @ q / scale
        order = np.argsort(-sims)[:5]
        # count as correct if the top hit is the same picture (same art)
        n += 1; top1 += int(order[0] == row); top5 += int(row in order)
    print(f"{mode}: top-1 {top1}/{n} = {100*top1/n:.0f}%   top-5 {top5}/{n} = {100*top5/n:.0f}%", flush=True)
