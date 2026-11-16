"""Price history from tcgcsv's daily archives: where each card's TCGplayer sales average stood 7 and 30 days
ago (the trend), and on how many of the last 14 days it moved (TCGplayer only recalculates it when copies
sell, so a number that never moves means nothing is selling)."""
import os, json, shutil, subprocess, tempfile, datetime, concurrent.futures as cf

ARCHIVE = "https://tcgcsv.com/archive/tcgplayer/prices-{day}.ppmd.7z"


def fetch_day(http, day, cats):
    """{(productId, subTypeName): marketPrice} for one past day, or None if that day's archive can't be had."""
    tmp = tempfile.mkdtemp(prefix="hist")
    try:
        raw = http(ARCHIVE.format(day=day), tries=3, timeout=120)
        if not raw or len(raw) < 10000:
            return None
        arc = os.path.join(tmp, "a.7z")
        with open(arc, "wb") as f:
            f.write(raw)
        r = subprocess.run(["7z", "x", "-y", f"-o{tmp}", arc] + [f"{day}/{c}/*" for c in cats], capture_output=True)
        if r.returncode != 0:
            return None
        out = {}
        for c in cats:
            root = os.path.join(tmp, day, str(c))
            if not os.path.isdir(root):
                continue
            for gid in os.listdir(root):
                try:
                    rows = json.load(open(os.path.join(root, gid, "prices")))["results"]
                except Exception:
                    continue
                for x in rows:
                    m = x.get("marketPrice")
                    if m is not None:
                        out[(x["productId"], x.get("subTypeName"))] = float(m)
        return out or None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_history(http, today, cats, days=30):
    """[(days_ago, {key: market})] for as many of the last [days] days as have an archive, newest first."""
    t = datetime.date.fromisoformat(today)
    ago = list(range(1, days + 1))
    with cf.ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda d: fetch_day(http, (t - datetime.timedelta(days=d)).isoformat(), cats), ago))
    return [(d, g) for d, g in zip(ago, got) if g]


def summarize(today_price, past):
    """past: {days_ago: price or None}. Returns [price 7 days ago, price 30 days ago, moves, pairs]:
    moves = on how many of the last 14 day-to-day steps the price changed; pairs = how many steps could be compared."""
    def near(target, lo, hi):
        best = None
        for d in range(lo, hi + 1):
            p = past.get(d)
            if p is not None and (best is None or abs(d - target) < abs(best[0] - target)):
                best = (d, p)
        return best[1] if best else None
    series = [today_price] + [past.get(d) for d in range(1, 15)]
    moves = pairs = 0
    for a, b in zip(series, series[1:]):
        if a is not None and b is not None:
            pairs += 1
            if abs(a - b) >= 0.005:
                moves += 1
    return [near(7, 5, 10), near(30, 24, 30), moves, pairs]


def apply(cards, history):
    """Append [p7, p30, moves, pairs] to every TCGplayer price that has a known source product. Returns count."""
    n = 0
    for c in cards:
        src = c.get("_src")
        if not src or not c.get("tp"):
            continue
        for k, key in src.items():
            v = c["tp"].get(k)
            if not v or len(v) != 3:
                continue
            past = {d: day.get(key) for d, day in history}
            s = summarize(v[0], past)
            if s[3] > 0 or s[0] is not None or s[1] is not None:
                c["tp"][k] = v + s
                n += 1
    return n


if __name__ == "__main__":
    # self-check
    assert summarize(10.0, {d: 10.0 for d in range(1, 31)}) == [10.0, 10.0, 0, 14]
    assert summarize(12.0, {d: 10.0 + (d % 2) for d in range(1, 31)}) == [11.0, 10.0, 14, 14]
    assert summarize(5.0, {}) == [None, None, 0, 0]
    assert summarize(5.0, {6: 4.0, 26: 3.0}) == [4.0, 3.0, 0, 0]
    assert summarize(None, {1: 2.0, 2: 2.0, 3: 2.5}) == [None, None, 1, 2]
    cards = [{"tp": {"holofoil": [10.0, 9.0, 11.0]}, "_src": {"holofoil": (1, "Holofoil")}}, {"tp": {"normal": [1, 1]}}]
    assert apply(cards, [(1, {(1, "Holofoil"): 10.0}), (7, {(1, "Holofoil"): 8.0})]) == 1
    assert cards[0]["tp"]["holofoil"] == [10.0, 9.0, 11.0, 8.0, None, 0, 1], cards[0]
    print("ok")
