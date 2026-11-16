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


def _fits(cn, pn):
    return bool(cn) and bool(pn) and (cn == pn or cn in pn or pn in cn)


def _holo(r):
    return "holo" in (r or "").lower()


def match_local(cards, groups, lang):
    """Japanese cards -> TCGplayer Japan products. Needs each card's English name ("en"): TCGplayer lists
    Japanese cards under English names. A set is paired with a TCGplayer set by its code ("SV9a: ..."), or by
    cards agreeing on number AND name, or — for old sets the two sources number differently — by sharing most
    of the same names. A card is then paired by number + name, else by a name only one card in each set has.
    Number alone is trusted only for cards with no English name (trainers) in a set paired by code or number."""
    mine = [(i, c) for i, c in enumerate(cards) if c.get("l", "en") == lang]
    sets = collections.defaultdict(list)
    for i, c in mine: sets[c["s"]].append((i, c))
    gidx = []
    for g in groups:
        by = collections.defaultdict(list); names = collections.Counter()
        for p in g["products"]:
            pn = nname(p.get("n")); names[pn] += 1
            n, t = nnum(p.get("no"))
            if n: by[n].append((t, pn, p))
        gidx.append((by, names))
    out = {}; stats = []
    for sid, cs in sets.items():
        cn_of = {i: nname(c.get("en") or "") for i, c in cs}
        named_cards = [i for i, c in cs if cn_of[i]]
        best = None
        for gi, (by, names) in enumerate(gidx):
            g = groups[gi]
            numbered = loose = 0
            left = collections.Counter(names)
            for i, c in cs:
                cn = cn_of[i]; n = nnum(c.get("no"))[0]
                if cn and any(_fits(cn, pn) for t, pn, p in by.get(n, ())): numbered += 1
                elif not cn and n in by: loose += 1
                if cn:
                    k = cn if left.get(cn, 0) > 0 else next((pn for pn in left if left[pn] > 0 and _fits(cn, pn)), None)
                    if k is not None: left[k] -= 1
            shared = sum(names.values()) - sum(left.values())
            code = (g["name"] or "").split(":")[0].strip().lower() == sid.lower()
            size_ok = 0.5 * len(cs) <= len(g["products"]) <= 2.5 * len(cs)
            how = None
            if code and (numbered >= 3 or numbered + loose >= max(4, 0.15 * len(cs))): how = "code"
            elif numbered >= max(4, 0.3 * len(named_cards)): how = "number"
            elif size_ok and shared >= max(8, 0.6 * len(named_cards)): how = "name"
            if how is None: continue
            rank = (2 if how == "code" else 1 if how == "number" else 0, numbered + 0.5 * shared)
            if best is None or rank > best[0]: best = (rank, gi, how)
        if best is None:
            stats.append((sid, cs[0][1].get("sn"), len(cs), None, 0)); continue
        _, gi, how = best
        by, names = gidx[gi]; g = groups[gi]; got = 0; used = set()
        same = collections.Counter((cn_of[i], _holo(c.get("r"))) for i, c in cs if cn_of[i])
        later = []
        for i, c in cs:
            cn = cn_of[i]; n = nnum(c.get("no"))[0]; cand = by.get(n, ())
            pick = [p for t, pn, p in cand if _fits(cn, pn)]
            if not pick and not cn and how != "name" and len(cand) == 1: pick = [cand[0][2]]
            if pick:
                out[i] = (g, pick[0]); used.add(pick[0]["id"]); got += 1
            elif cn: later.append((i, c))
        for i, c in later:                               # by name alone, when it can only be one product
            cn = cn_of[i]
            if same[(cn, _holo(c.get("r")))] != 1: continue
            ps = [p for p in g["products"] if nname(p.get("n")) == cn] or [p for p in g["products"] if _fits(cn, nname(p.get("n")))]
            ps = [p for p in ps if p["id"] not in used]
            if len(ps) > 1:
                ps = [p for p in ps if _holo(p.get("r")) == _holo(c.get("r"))]
            if len(ps) == 1:
                out[i] = (g, ps[0]); used.add(ps[0]["id"]); got += 1
        stats.append((sid, cs[0][1].get("sn"), len(cs), g["name"] + f" [{how}]", got))
    return out, stats


def apply(cards, groups, lang, today):
    """Add TCGplayer pictures / prices / links to [cards] of one language. Returns (pictures, prices) added."""
    m, stats = match_singles(cards, groups, lang) if lang == "en" else match_local(cards, groups, lang)
    pics = prices = 0
    for i, (g, p) in m.items():
        c = cards[i]
        if p.get("u"): c["u"] = p["u"]
        if lang != "en" and c.get("en"):
            # TCGplayer's name is fuller than the one worked out from the Pokédex number ("Dark Houndoom", "Shining Charizard").
            full = re.sub(r"\(.*?\)|\[.*?\]", " ", re.sub(r"\s*-\s*[\w]+/[\w]+\s*$", "", p.get("n") or "")).strip()
            full = re.sub(r"\s+", " ", full)
            if full and nname(full) != nname(c["en"]) and nname(c["en"]) in nname(full) and len(full) <= 40: c["en"] = full
        if not c.get("img") and p.get("img"):
            c["img"] = p["img"]; pics += 1
        # Prices straight from TCGplayer's list for today: [recent-sales average, cheapest listing, typical listing].
        # These replace the main source's copy (same numbers, but that copy can be days old and has no typical listing).
        tp = {}; src = {}
        for x in g["prices"]:
            if x["id"] != p["id"]: continue
            k = SUB.get(x.get("t") or "")
            if k and (x.get("m") is not None or x.get("lo") is not None or x.get("mid") is not None):
                tp[k] = [x.get("m"), x.get("lo"), x.get("mid")]
                src[k] = (x["id"], x.get("t"))      # which TCGplayer price row this is, for price history
        if tp:
            c["tp"] = tp; c["pu"] = today; prices += 1
            c["_src"] = src                         # build-time only; removed before the catalog is written
    return pics, prices, stats
