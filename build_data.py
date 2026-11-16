"""
CardSight daily data builder.

Collects every Pokemon card in English, Japanese, Chinese (Traditional) and Chinese (Simplified)
from TCGdex (names, set, number, rarity, variants, TCGplayer + Cardmarket prices), then builds
compact "picture fingerprints" (image embeddings) so the phone can recognise a card by its art.

Outputs (in ./out):
  catalog.json.gz  every card + prices
  emb.bin          fingerprints, PCA-compressed to int8
  embedder.tflite  the image model the phone must use (same one used here)
  embcache.npz     full-size fingerprints, reused next run so only new cards are processed
  manifest.json    version + file hashes (the phone checks this first)
"""
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

API = "https://api.tcgdex.net/v2"
LANGS = ["en", "ja", "zh-tw", "zh-cn"]
ASIAN = ["ja", "zh-tw", "zh-cn"]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_embedder/"
             "mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite")
PCA_DIMS = 256
FP_FORMAT = "full+art/2"  # bump when fingerprints are made differently, to rebuild the cache
OUT = "out"
CACHE = "cache"
UA = {"User-Agent": "CardSight-data/1.0 (+https://github.com/naxrabot/cardsight-data)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http(url, tries=8, timeout=60):
    """GET with retries. Returns bytes, or None on 404."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(min(2 ** attempt, 30))
    return None


def num(v):
    return round(float(v), 2) if isinstance(v, (int, float)) else None


def compact_card(lang, c):
    s = c.get("set") or {}
    cc = s.get("cardCount") or {}
    v = c.get("variants") or {}
    flags = "".join(k for k, on in (("N", v.get("normal")), ("R", v.get("reverse")),
                                    ("H", v.get("holo")), ("F", v.get("firstEdition"))) if on)
    dex = c.get("dexId") or []
    rec = {
        "i": c["id"], "l": lang, "n": c.get("name") or "", "no": c.get("localId") or "",
        "s": s.get("id") or "", "sn": s.get("name") or "",
        "so": int(cc.get("official") or 0), "st": int(cc.get("total") or 0),
        "r": "" if c.get("rarity") in (None, "None") else c["rarity"],
        "v": flags, "d": int(dex[0]) if dex and isinstance(dex[0], (int, float)) else 0, "img": c.get("image"),
    }
    pr = c.get("pricing") or {}
    tp = pr.get("tcgplayer") or {}
    tpo = {}
    for k, val in tp.items():
        if isinstance(val, dict):
            m, lo = num(val.get("marketPrice")), num(val.get("lowPrice"))
            if m is not None or lo is not None:
                tpo[k] = [m, lo]
    if tpo:
        rec["tp"] = tpo
    cm = pr.get("cardmarket") or {}
    if cm:
        vals = [num(cm.get(k)) for k in ("avg1", "avg7", "avg30", "trend",
                                         "avg1-holo", "avg7-holo", "avg30-holo", "trend-holo")]
        if any(x for x in vals):
            rec["cm"] = vals
    upd = (tp.get("updated") or cm.get("updated") or "")[:10]
    if upd:
        rec["pu"] = upd
    return rec


def img_url(u):
    """Download address for a catalog picture: TCGdex pictures are a base address, TCGplayer's are complete."""
    if u.endswith((".jpg", ".png", ".webp")):
        return u.replace("_200w.", "_400w.")
    return u + "/low.png"


# Pokémon TCG Pocket (the phone game) — digital-only cards; they are never sold on a stream and only
# cause wrong matches.
POCKET_SET = re.compile(r"^([AB]\d[ab]?|P-[A-Z])$")


def fetch_singles():
    """TCGplayer's list of single cards (English + Japanese) with pictures and prices."""
    out = {}
    for cat, lang in SEALED_CATEGORIES:
        groups = json.loads(http(f"{TCGCSV}/{cat}/groups"))["results"]

        def one(g):
            gid = g["groupId"]
            try:
                prods = json.loads(http(f"{TCGCSV}/{cat}/{gid}/products", tries=4) or b'{"results":[]}')["results"]
                prices = json.loads(http(f"{TCGCSV}/{cat}/{gid}/prices", tries=4) or b'{"results":[]}')["results"]
            except Exception:
                return None
            ps = []
            for p in prods:
                ext = {e.get("name"): e.get("value") for e in (p.get("extendedData") or [])}
                if "Number" not in ext and "Rarity" not in ext and "HP" not in ext:
                    continue
                ps.append({"id": p["productId"], "n": p.get("name"), "img": p.get("imageUrl"), "u": p.get("url"), "no": ext.get("Number")})
            pr = [{"id": x["productId"], "t": x.get("subTypeName"), "m": num(x.get("marketPrice")), "lo": num(x.get("lowPrice")),
                   "mid": num(x.get("midPrice"))} for x in prices]
            return {"id": gid, "name": g.get("name"), "products": ps, "prices": pr}

        with cf.ThreadPoolExecutor(8) as ex:
            out[lang] = [g for g in ex.map(one, groups) if g]
        log(f"TCGplayer singles [{lang}]: {sum(len(g['products']) for g in out[lang])} cards in {len(out[lang])} sets")
    return out


def fetch_lang(lang):
    raw = http(f"{API}/{lang}/cards")
    brief = json.loads(raw)
    log(f"[{lang}] {len(brief)} cards listed")
    out, failed = [], 0

    def one(b):
        d = http(f"{API}/{lang}/cards/{urllib.request.quote(b['id'], safe='')}")
        return json.loads(d) if d else None

    with cf.ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(one, b): b for b in brief}
        for n, f in enumerate(cf.as_completed(futs), 1):
            b = futs[f]
            try:
                c = f.result()
            except Exception:
                c = None
            if c is None:
                failed += 1
                c = {"id": b["id"], "localId": b.get("localId"), "name": b.get("name"), "image": b.get("image")}
            out.append(compact_card(lang, c))
            if n % 3000 == 0:
                log(f"[{lang}] {n}/{len(brief)}")
    log(f"[{lang}] done, {failed} without details")
    return out


# ---------------------------------------------------------------- sealed products (packs, boxes, ETBs, cases...)

TCGCSV = "https://tcgcsv.com/tcgplayer"
SEALED_CATEGORIES = [(3, "en"), (85, "ja")]  # TCGplayer: Pokemon, Pokemon Japan


def sealed_kind(name):
    n = name.lower()
    if "case" in n or "display" in n and "booster box" not in n and "booster display" not in n:
        return "Case" if "case" in n else "Display"
    if "booster box" in n or "booster display" in n or "display box" in n:
        return "Booster Box"
    if "elite trainer box" in n or re.search(r"\betb\b", n):
        return "Elite Trainer Box"
    if "booster bundle" in n:
        return "Booster Bundle"
    if "blister" in n:
        return "Blister"
    if "build & battle" in n or "build and battle" in n:
        return "Build & Battle"
    if re.search(r"\btin\b", n):
        return "Tin"
    if "booster pack" in n or "sleeved booster" in n or re.search(r"\bpack\b", n):
        return "Booster Pack"
    if "deck" in n:
        return "Deck"
    if "collection" in n or "premium" in n or re.search(r"\bbox\b", n):
        return "Collection Box"
    return "Sealed"


def fetch_sealed():
    """Every sealed product TCGplayer lists for Pokemon (English + Japanese), with prices (USD)."""
    out = []
    for cat, lang in SEALED_CATEGORIES:
        groups = json.loads(http(f"{TCGCSV}/{cat}/groups"))["results"]

        def one(g):
            gid = g["groupId"]
            try:
                prods = json.loads(http(f"{TCGCSV}/{cat}/{gid}/products", tries=4) or b'{"results":[]}')["results"]
                prices = json.loads(http(f"{TCGCSV}/{cat}/{gid}/prices", tries=4) or b'{"results":[]}')["results"]
            except Exception:
                return []
            pmap = {}
            for p in prices:
                pmap.setdefault(p["productId"], []).append(p)
            items = []
            for p in prods:
                ext = {e.get("name") for e in (p.get("extendedData") or [])}
                # Single cards carry card details (number, rarity, HP, attacks…); sealed products don't.
                if ext & {"Number", "Rarity", "Card Type", "CardType", "HP", "Stage", "Attack 1", "Weakness", "RetreatCost"}:
                    continue
                name = p.get("name") or ""
                if "code card" in name.lower():
                    continue
                pr = pmap.get(p["productId"], [])
                best = max(pr, key=lambda x: x.get("marketPrice") or 0, default=None)
                if not best or not (best.get("marketPrice") or best.get("lowPrice")):
                    continue                  # no price data at all
                items.append({
                    "i": str(p["productId"]), "l": lang, "n": name, "s": g.get("name") or "",
                    "k": sealed_kind(name), "img": p.get("imageUrl"), "u": p.get("url"),
                    "m": num(best.get("marketPrice")), "lo": num(best.get("lowPrice")),
                    "mid": num(best.get("midPrice")), "hi": num(best.get("highPrice")),
                    "d": (g.get("publishedOn") or "")[:10],
                })
            return items

        with cf.ThreadPoolExecutor(8) as ex:
            for items in ex.map(one, groups):
                out.extend(items)
        log(f"sealed [{lang}]: {sum(1 for x in out if x['l'] == lang)} products from {len(groups)} sets")
    return out


# ---------------------------------------------------------------- fingerprints

def make_embedder(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.ImageEmbedderOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path), l2_normalize=True)
    emb = vision.ImageEmbedder.create_from_options(opts)

    def run(pil):
        arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
        return np.asarray(emb.embed(img).embeddings[0].embedding, dtype=np.float32)
    return run


def degrade(pil):
    """Imitate a livestream/phone-photo view of a card: low-res, JPEG-blocky, a bit dark and soft."""
    from PIL import Image, ImageEnhance, ImageFilter
    w, h = pil.size
    small = pil.resize((max(60, w // 3), max(84, h // 3)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    small = ImageEnhance.Brightness(small).enhance(0.85).filter(ImageFilter.GaussianBlur(0.8))
    buf = io.BytesIO()
    small.convert("RGB").save(buf, "JPEG", quality=35)
    return Image.open(io.BytesIO(buf.getvalue()))


def art_box(pil):
    """The artwork window of a card (same proportions the phone uses)."""
    w, h = pil.size
    return pil.crop((int(w * .07), int(h * .10), int(w * .93), int(h * .53)))


def pca_int8(X):
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = vt[:PCA_DIMS].T.astype(np.float32)          # D x K
    P = (X - mean) @ comps
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    scale = 127.0 / float(np.percentile(np.abs(P), 99.9))
    Q = np.clip(np.round(P * scale), -127, 127).astype(np.int8)
    return mean.astype("<f4"), comps.astype("<f4"), scale, Q


def build_fingerprints(cards, model_path):
    from PIL import Image
    # Asian printings often lack an image in one language but share the exact art (same id) in another.
    by_id = {}
    for c in cards:
        if c["l"] in ASIAN and c.get("img"):
            by_id.setdefault(c["i"], c["img"])
    for c in cards:
        if not c.get("img") and c["l"] in ASIAN and c["i"] in by_id:
            c["img"] = by_id[c["i"]]
            c["ib"] = 1  # image borrowed from another language

    urls = sorted({c["img"] for c in cards if c.get("img")})
    cache = {}
    msha = sha(model_path)
    cpath = os.path.join(CACHE, "embcache.npz")
    if os.path.exists(cpath):
        try:
            z = np.load(cpath, allow_pickle=False)
            if str(z["model"]) != msha + FP_FORMAT:
                raise ValueError("image model or fingerprint format changed")
            for u, v, a in zip(z["urls"], z["vecs"], z["arts"]):
                cache[str(u)] = (v.astype(np.float32), a.astype(np.float32))
            log(f"fingerprint cache: {len(cache)} reused")
        except Exception as e:
            log("cache unreadable, rebuilding:", e)

    todo = [u for u in urls if u not in cache]
    log(f"fingerprints: {len(urls)} images, {len(todo)} new")
    embed = make_embedder(model_path)

    def fetch(u):
        try:
            return u, http(img_url(u), tries=6, timeout=30)
        except Exception:
            return u, None  # skip this picture today; text matching still covers the card

    with cf.ThreadPoolExecutor(12) as ex:
        for n, (u, b) in enumerate(ex.map(fetch, todo), 1):
            if b:
                try:
                    pil = Image.open(io.BytesIO(b)).convert("RGB")
                    art = art_box(pil)
                    # clean + "livestream-quality" views averaged: much more robust to blur/glare
                    v = embed(pil) + embed(degrade(pil))
                    a = embed(art) + embed(degrade(art))
                    cache[u] = ((v / (np.linalg.norm(v) + 1e-9)).astype(np.float32),
                                (a / (np.linalg.norm(a) + 1e-9)).astype(np.float32))
                except Exception as e:
                    log("embed failed", u, e)
            if n % 2000 == 0:
                log(f"fingerprints {n}/{len(todo)}")

    good = [u for u in urls if u in cache]
    np.savez_compressed(os.path.join(OUT, "embcache.npz"), model=np.array(msha + FP_FORMAT),
                        urls=np.array(good),
                        vecs=np.stack([cache[u][0] for u in good]).astype(np.float16),
                        arts=np.stack([cache[u][1] for u in good]).astype(np.float16))
    row = {u: i for i, u in enumerate(good)}
    for c in cards:
        c["e"] = row.get(c.get("img"), -1)

    XF = np.stack([cache[u][0] for u in good]).astype(np.float32)
    XA = np.stack([cache[u][1] for u in good]).astype(np.float32)
    mF, cF, sF, QF = pca_int8(XF)
    mA, cA, sA, QA = pca_int8(XA)
    D, K, N = XF.shape[1], PCA_DIMS, len(good)
    with open(os.path.join(OUT, "emb.bin"), "wb") as f:
        f.write(b"CSE2")
        f.write(struct.pack("<iiiff", D, K, N, sF, sA))
        for arr in (mF, cF, mA, cA):
            f.write(arr.tobytes())
        f.write(QF.tobytes())
        f.write(QA.tobytes())
    log(f"emb.bin: N={N} D={D} K={K}")
    return N, good


NET_W, NET_H = 192, 256


def net_views(png_bytes):
    """A catalog picture as the trained model saw it in training: clean, plus a small soft copy."""
    import cv2
    c = cv2.cvtColor(cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    c = cv2.resize(c, (245, 337), interpolation=cv2.INTER_AREA)
    a = cv2.resize(c, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    b = cv2.resize(cv2.resize(c, (82, 112), interpolation=cv2.INTER_AREA), (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    return np.stack([a, b])


def build_net_fingerprints(good, fetch_png=None):
    """Fingerprints from CardSight's own livestream-trained picture model (emb3.bin + cardnet.onnx).
    Rows line up with emb.bin / the catalog's "e" numbers. Skipped until a trained model is published."""
    model = os.path.join(CACHE, "cardnet.onnx")
    if not os.path.exists(model):
        log("trained picture model: none published yet")
        return False
    import onnxruntime as ort
    msha = sha(model)
    cache = {}
    for name, need_sha in (("netcache.npz", True), ("gallery_seed.npz", False)):
        cpath = os.path.join(CACHE, name)
        if cache or not os.path.exists(cpath):
            continue
        try:
            z = np.load(cpath, allow_pickle=False)
            if need_sha and str(z["model"]) != msha:
                raise ValueError("picture model changed")
            for u, v in zip(z["urls"], z["vecs"]):
                cache[str(u)] = v.astype(np.float32)
            log(f"trained fingerprints: {len(cache)} reused from {name}")
        except Exception as e:
            log(name, "not used:", e)
    todo = [u for u in good if u not in cache]
    log(f"trained fingerprints: {len(good)} pictures, {len(todo)} new")
    if todo:
        sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name

        def fetch(u):
            try:
                b = fetch_png(u) if fetch_png else http(img_url(u), tries=6, timeout=30)
                return u, net_views(b) if b else None
            except Exception:
                return u, None

        def flush(us, vs):
            if not us:
                return
            x = np.concatenate(vs).transpose(0, 3, 1, 2).astype(np.float32)
            e = sess.run(None, {inp: x})[0].reshape(len(us), 2, -1).sum(1)
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
            for u, v in zip(us, e):
                cache[u] = v.astype(np.float32)

        with cf.ThreadPoolExecutor(12) as ex:
            us, vs = [], []
            for n, (u, v) in enumerate(ex.map(fetch, todo), 1):
                if v is not None:
                    us.append(u)
                    vs.append(v)
                if len(us) >= 64:
                    flush(us, vs)
                    us, vs = [], []
                if n % 2000 == 0:
                    log(f"trained fingerprints {n}/{len(todo)}")
            flush(us, vs)
    have = [u for u in good if u in cache]
    if not have:
        return False
    K = len(cache[have[0]])
    np.savez_compressed(os.path.join(OUT, "netcache.npz"), model=np.array(msha), urls=np.array(have),
                        vecs=np.stack([cache[u] for u in have]).astype(np.float16))
    zero = np.zeros(K, np.float32)
    X = np.stack([cache.get(u, zero) for u in good]).astype(np.float32)
    scale = 127.0 / float(np.percentile(np.abs(X), 99.9))
    Q = np.clip(np.round(X * scale), -127, 127).astype(np.int8)
    with open(os.path.join(OUT, "emb3.bin"), "wb") as f:
        f.write(b"CSE3")
        f.write(struct.pack("<iif", K, len(good), scale))
        f.write(Q.tobytes())
    shutil.copyfile(model, os.path.join(OUT, "cardnet.onnx"))
    log(f"emb3.bin: N={len(good)} K={K} ({len(good) - len(have)} without a picture)")
    return True


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    cards = []
    counts = {}
    for lang in LANGS:
        lc = fetch_lang(lang)
        counts[lang] = len(lc)
        cards.extend(lc)
    if counts.get("en", 0) < 15000:
        sys.exit(f"English catalog too small ({counts}); not publishing")

    before = len(cards)
    cards = [c for c in cards if not POCKET_SET.match(c["s"])]
    log(f"left out {before - len(cards)} TCG Pocket (digital-only) cards")

    # Pictures and USD prices from TCGplayer for cards the main source has none for.
    try:
        import tcg_singles
        singles = fetch_singles()
        today = time.strftime("%Y-%m-%d", time.gmtime())
        for lang in ("en", "ja"):
            if sum(len(g["products"]) for g in singles.get(lang, [])) < 5000:
                log(f"TCGplayer singles [{lang}]: too few, skipped")
                continue
            pics, prices, stats = tcg_singles.apply(cards, singles[lang], lang, today)
            log(f"TCGplayer [{lang}]: +{pics} pictures, {prices} cards priced from today's TCGplayer list; "
                f"{sum(1 for s_ in stats if s_[3] is None)} of {len(stats)} sets not found")
    except Exception as e:
        log("TCGplayer singles skipped:", e)
    # Price history: where each price stood 7 and 30 days ago, and how often it has moved (= how often it sells).
    try:
        import price_history
        if any("_src" in c for c in cards):
            hist = price_history.fetch_history(http, time.strftime("%Y-%m-%d", time.gmtime()), [cat for cat, _ in SEALED_CATEGORIES])
            n = price_history.apply(cards, hist)
            log(f"price history: {len(hist)} past days, {n} prices given a trend")
    except Exception as e:
        log("price history skipped:", e)
    for c in cards:
        c.pop("_src", None)

    # English names for Japanese/Chinese cards (shown in the app, and matched against English listing titles).
    try:
        from en_names import Namer
        namer = Namer()
        named = 0
        for c in cards:
            if c["l"] != "en":
                e = namer.english(c["n"], c["d"])
                if e:
                    c["en"] = e
                    named += 1
        log(f"English names: {named}/{sum(1 for c in cards if c['l'] != 'en')} Japanese/Chinese cards")
    except Exception as e:
        log("English names skipped:", e)

    model_path = os.path.join(OUT, "embedder.tflite")
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(http(MODEL_URL))
    n_emb, good = build_fingerprints(cards, model_path)
    trained = False
    try:
        trained = build_net_fingerprints(good)
    except Exception as e:
        log("trained fingerprints skipped:", e)

    # The phone expects exact types; refuse to publish anything else.
    for c in cards:
        for k in ("so", "st", "d", "e"):
            if not isinstance(c.get(k), int):
                sys.exit(f"bad {k}={c.get(k)!r} in {c['i']} ({c['l']}); not publishing")
        for k in ("i", "l", "n", "no", "s", "sn", "r", "v"):
            if not isinstance(c.get(k), str):
                sys.exit(f"bad {k}={c.get(k)!r} in {c['i']}; not publishing")

    # EUR -> USD so Cardmarket-only cards can be valued in dollars.
    eur_usd = 1.08
    try:
        rate = json.loads(http("https://api.frankfurter.app/latest?from=EUR&to=USD", tries=3))
        eur_usd = float(rate["rates"]["USD"])
    except Exception as e:
        log("exchange rate unavailable, using default:", e)

    # Price history: highest market price recorded for each card and printing, kept from day to day.
    today = time.strftime("%Y-%m-%d", time.gmtime())
    hist = {}
    hpath = os.path.join(CACHE, "history.json.gz")
    if os.path.exists(hpath):
        try:
            with gzip.open(hpath, "rt", encoding="utf-8") as f:
                hist = json.load(f)
        except Exception as e:
            log("history unreadable, starting fresh:", e)
    for c in cards:
        key = c["l"] + ":" + c["i"]
        h = hist.setdefault(key, {})
        prices = {k: v[0] for k, v in (c.get("tp") or {}).items() if v and v[0]}
        cm = c.get("cm")
        if cm and cm[3]:
            prices["cm"] = round(cm[3] * eur_usd, 2)       # Cardmarket trend, in USD
        if cm and len(cm) > 7 and cm[7]:
            prices["cm-foil"] = round(cm[7] * eur_usd, 2)
        for k, p in prices.items():
            if k not in h or p > h[k][0]:
                h[k] = [p, today]
        if h:
            c["hi"] = h
        else:
            hist.pop(key, None)

    # Sealed products, with the same "highest recorded" history as cards.
    sealed = []
    try:
        sealed = fetch_sealed()
    except Exception as e:
        log("sealed products skipped:", e)
    for p in sealed:
        h = hist.setdefault("sealed:" + p["i"], {})
        if p.get("m"):
            if "m" not in h or p["m"] > h["m"][0]:
                h["m"] = [p["m"], today]
            p["hx"] = h["m"]
    with gzip.open(os.path.join(OUT, "history.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(hist, f, separators=(",", ":"))
    with gzip.open(os.path.join(OUT, "sealed.json.gz"), "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump({"products": sealed}, f, ensure_ascii=False, separators=(",", ":"))

    version = int(time.time())
    with gzip.open(os.path.join(OUT, "catalog.json.gz"), "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump({"version": version, "cards": cards}, f, ensure_ascii=False, separators=(",", ":"))

    files = {}
    names = ["catalog.json.gz", "emb.bin", "embedder.tflite", "sealed.json.gz"]
    if trained:
        names += ["emb3.bin", "cardnet.onnx"]
    for name in names:
        p = os.path.join(OUT, name)
        files[name] = {"sha256": sha(p), "size": os.path.getsize(p)}
    manifest = {
        "version": version,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts, "fingerprints": n_emb, "files": files, "eur_usd": eur_usd,
        "sealed": len(sealed),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log("manifest:", json.dumps(manifest))


if __name__ == "__main__":
    main()
