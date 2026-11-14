"""Fills gaps in the catalog from TCGplayer's own card list (via tcgcsv.com): pictures and USD prices for
cards the main source has none for — trainer galleries, promos, McDonald's sets, most Japanese cards."""
import re, collections, unicodedata
def nnum(s):
    """'003/103' -> ('3','103'); 'TG17/TG30' -> ('TG17','TG30'); 'SWSH123' -> ('SWSH123','')"""
    s = (s or "").strip().upper().replace(" ", "")
    a, _, b = s.partition("/")
    f = lambda x: (re.sub(r"^0+(?=\d)", "", x) if x.isdigit() else re.sub(r"(?<=\D)0+(?=\d)", "", x))
    return f(a), f(b)
def nname(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\s*-\s*[\w]+/[\w]+\s*$", "", s)          # "Masquerain - 003/103"
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)                # "(Full Art)", "(Secret)"
    return re.sub(r"[^a-z0-9]+", "", s)
SUB = {"Normal": "normal", "Holofoil": "holofoil", "Reverse Holofoil": "reverse-holofoil", "1st Edition": "1st-edition",
       "1st Edition Holofoil": "1st-edition-holofoil", "Unlimited": "unlimited", "Unlimited Holofoil": "unlimited-holofoil"}
def match_singles(cards, groups, lang):
    """groups: tcgcsv groups (with products/prices) for one language. Returns {card index: product}, stats."""
    mine = [(i, c) for i, c in enumerate(cards) if c.get("l", "en") == lang]
    sets = collections.defaultdict(list)
    for i, c in mine: sets[c["s"]].append((i, c))
    # index every group's products by number
    gidx = []
    for g in groups:
        by = collections.defaultdict(list)
        for p in g["products"]:
            n, t = nnum(p.get("no"))
            if n: by[n].append((t, nname(p.get("n")), p))
        gidx.append(by)
    out = {}; stats = []
    for sid, cs in sets.items():
        def name_of(c): return nname(c.get("n") if lang == "en" else (c.get("en") or ""))
        best = None
        for gi, by in enumerate(gidx):
            hit = named = 0
            for _, c in cs:
                n = nnum(c.get("no"))[0]
                for t, pn, p in by.get(n, ()):
                    cn = name_of(c)
                    if cn and (cn == pn or cn in pn or pn in cn): named += 1; hit += 1; break
                else:
                    if n in by and not name_of(c): hit += 0.3      # no English name to check (trainers etc.)
            if lang != "en":
                pre = (groups[gi]["name"] or "").split(":")[0].strip().lower()
                if pre == sid.lower(): hit += 5                      # "SV9a: Heat Wave Arena" is set SV9a
            if best is None or hit > best[0]: best = (hit, named, gi)
        hit, named, gi = best
        need = max(3, 0.25 * len(cs)) if lang == "en" else max(4, 0.15 * len(cs))
        if hit < need:
            stats.append((sid, cs[0][1].get("sn"), len(cs), None, hit)); continue
        by = gidx[gi]; got = 0
        for i, c in cs:
            n = nnum(c.get("no"))[0]; cn = name_of(c); cand = by.get(n, ())
            pick = [p for t, pn, p in cand if cn and (cn == pn or cn in pn or pn in cn)]
            if not pick and lang != "en" and not cn and len(cand) == 1: pick = [cand[0][2]]
            if len(pick) >= 1:
                out[i] = (groups[gi], pick[0]); got += 1
        stats.append((sid, cs[0][1].get("sn"), len(cs), groups[gi]["name"], got))
    return out, stats


def apply(cards, groups, lang, today):
    """Add TCGplayer pictures / prices / links to [cards] of one language. Returns (pictures, prices) added."""
    m, stats = match_singles(cards, groups, lang)
    pics = prices = 0
    for i, (g, p) in m.items():
        c = cards[i]
        if p.get("u"): c["u"] = p["u"]
        if not c.get("img") and p.get("img"):
            c["img"] = p["img"]; pics += 1
        # Prices straight from TCGplayer's list for today: [recent-sales average, cheapest listing, typical listing].
        # These replace the main source's copy (same numbers, but that copy can be days old and has no typical listing).
        tp = {}
        for x in g["prices"]:
            if x["id"] != p["id"]: continue
            k = SUB.get(x.get("t") or "")
            if k and (x.get("m") is not None or x.get("lo") is not None or x.get("mid") is not None):
                tp[k] = [x.get("m"), x.get("lo"), x.get("mid")]
        if tp:
            c["tp"] = tp; c["pu"] = today; prices += 1
    return pics, prices, stats
