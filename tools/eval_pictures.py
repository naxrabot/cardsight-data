"""Compare picture-matching approaches on fake 'livestream' views (gallery = 3000 random cards)."""
import gzip, io, json, random, sys, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
sys.path.insert(0, ".")
from build_data import make_embedder, http, degrade

BASE = "https://github.com/naxrabot/cardsight-data/releases/download/latest/"
open("m.tflite", "wb").write(urllib.request.urlopen(BASE + "embedder.tflite").read())
cat = json.loads(gzip.decompress(urllib.request.urlopen(BASE + "catalog.json.gz").read()))["cards"]
embed = make_embedder("m.tflite")
urls = sorted({c["img"] for c in cat if c.get("img") and c["l"] == "en"})
random.seed(7)
gallery = random.sample(urls, 3000)
queries = gallery[:200]

def fetch(u):
    try:
        return u, http(u + "/high.png", tries=3, timeout=30)
    except Exception:
        return u, None
imgs = {}
with cf.ThreadPoolExecutor(12) as ex:
    for u, b in ex.map(fetch, gallery):
        if b: imgs[u] = Image.open(io.BytesIO(b)).convert("RGB").resize((367, 512), Image.BILINEAR)
gallery = [u for u in gallery if u in imgs]; queries = [u for u in queries if u in imgs]
print("gallery", len(gallery), "queries", len(queries), flush=True)

def art(im):  # artwork box of a modern card layout
    w, h = im.size
    return im.crop((int(w * .07), int(h * .10), int(w * .93), int(h * .53)))

def norm(v): return v / (np.linalg.norm(v) + 1e-9)

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
    m = random.uniform(0.0, 0.08)
    l, t = (out.width - cw) / 2 - cw * m, (out.height - ch) / 2 - ch * m
    return out.crop((int(l), int(t), int(l + cw * (1 + 2 * m)), int(t + ch * (1 + 2 * m))))

# reference fingerprints
G_full = np.stack([norm(embed(imgs[u]) + embed(degrade(imgs[u]))) for u in gallery])
G_full1 = np.stack([norm(embed(imgs[u])) for u in gallery])
G_art = np.stack([norm(embed(art(imgs[u])) + embed(degrade(art(imgs[u])))) for u in gallery])
print("refs done", flush=True)
idx = {u: i for i, u in enumerate(gallery)}

for px in (300, 200):
    res = {k: [0, 0] for k in ("full", "full_noaug", "art", "full+art")}
    for u in queries:
        q = stream_view(imgs[u], px)
        # the app's card finder trims most background; emulate a tight-ish crop
        qf = norm(embed(q)); qa = norm(embed(art(q)))
        for name, sims in (("full", G_full @ qf), ("full_noaug", G_full1 @ qf), ("art", G_art @ qa),
                           ("full+art", G_full @ qf + G_art @ qa)):
            o = np.argsort(-sims)[:5]
            res[name][0] += int(o[0] == idx[u]); res[name][1] += int(idx[u] in o)
    n = len(queries)
    print(f"card {px}px wide: " + "  ".join(f"{k}: top1 {100*a/n:.0f}% top5 {100*b/n:.0f}%" for k, (a, b) in res.items()), flush=True)
