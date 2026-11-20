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
# The same card printed in a special foil pattern or with a stamp. TCGplayer lists each as its own product
# ("Klink (Master Ball Pattern)") with its own price — often many times the plain card's.
# Checked in this order: the first word found names the style.
STYLE_KEYS = [("master ball", "master-ball"), ("poke ball", "poke-ball"), ("energy symbol", "energy-symbol"),
              ("dusk ball", "dusk-ball"), ("love ball", "love-ball"), ("friend ball", "friend-ball"),
              ("quick ball", "quick-ball"), ("team rocket pattern", "team-rocket"), ("terastal pattern", "terastal"),
              ("cosmo", "cosmos-holo"), ("cracked ice", "cracked-ice"),
              ("web holo", "special-holo"), ("line holo", "special-holo"), ("sheen holo", "special-holo"),
              ("mirage holo", "special-holo"), ("sparkle holo", "special-holo"), ("tinsel holo", "special-holo"),
              ("galaxy holo", "special-holo"), ("confetti holo", "special-holo"), ("pixel holo", "special-holo"),
              ("mirror", "mirror-holo"),                    # Japanese "Mirror Holofoil / Mirror Holo / Mirror Foil"
              ("staff", "staff"), ("prerelease", "prerelease"),
              ("1st place", "trophy"), ("2nd place", "trophy"), ("3rd place", "trophy"), ("4th place", "trophy"),
              ("winner", "trophy"), ("champion]", "trophy"), ("finalist", "trophy"), ("top 8", "trophy"), ("top 16", "trophy"),
              ("prize pack", "prize-pack"), ("holiday calendar", "holiday-calendar"), ("pokemon center", "pokemon-center"),
              ("league", "league"), ("regional championship", "league"), ("city championship", "league"),
              ("state championship", "league"), ("national championship", "league"), ("international championship", "league")]
# A bracket that only says which ordinary finish this listing is: not a style, it fills that finish's price.
FINISH_LABELS = {"non-holo": "normal", "non holo": "normal", "reverse holo": "reverse-holofoil",
                 "reverse holofoil": "reverse-holofoil"}
# Catch-all TCGplayer sets whose every card is a stamped / special version of a card from a real set.
GROUP_STYLE = {"prize pack series cards": "prize-pack", "league & championship cards": "league"}
# Away from the card's own set a bracket is often just the set's name ("(EX Team Magma vs Team Aqua)" on a deck
# reprint). There it only counts as a stamp when it says so.
STAMP_WORDS = ("stamp", "exclusive", "promo", "celebration", "pokemon day", "anniversary", "gym", "holo", "toys r us")


def style_of(name, group="", strict=False):
    """'' for a plain listing, else which special style a product name's brackets describe ("stamped" = any other
    stamp). "=normal" / "=reverse-holofoil" when the bracket only names an ordinary finish."""
    labels = [l.strip().lower() for l in re.findall(r"[\(\[]([^)\]]+)[\)\]]", name or "")]
    labels = [l for l in labels if not re.fullmatch(r"[\d/ ]+", l)]          # "(15)" is just the card number
    if not labels:
        return GROUP_STYLE.get((group or "").strip().lower(), "")
    if len(labels) == 1 and labels[0] in FINISH_LABELS:
        return "=" + FINISH_LABELS[labels[0]]
    joined = " ".join(f"[{l}]" for l in labels)
    for word, key in STYLE_KEYS:
        if word in joined:
            return key
    if strict and not any(w in joined for w in STAMP_WORDS):
        return GROUP_STYLE.get((group or "").strip().lower(), "")
    return GROUP_STYLE.get((group or "").strip().lower(), "stamped")


def split_styles(pick):
    """From the products sharing a card's number and name: (the plain one, {style: product})."""
    if len(pick) == 1:
        return pick[0], {}
    tagged = [(style_of(p.get("n")), p) for p in pick]
    plain = [p for st, p in tagged if not st]
    if not plain:
        # only special printings exist (promos): the first non-staff one stands for the card
        base = next((p for st, p in tagged if st != "staff"), tagged[0][1])
    else:
        base = plain[0]
    styles = {}
    for st, p in tagged:
        if p is not base and st and st not in styles:
            styles[st] = p
    return base, styles


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
                # Decks reprinted for the World Championships list one card under several players' names — not styles.
                if "world championship" in (groups[gi]["name"] or "").lower():
                    base, styles = pick[0], {}
                else:
                    base, styles = split_styles(pick)
                out[i] = (groups[gi], base, styles); got += 1
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
                # "Oddish" and "Oddish (Mirror Holofoil)" share a number: the plain one is the card, the rest are styles.
                base, styles = split_styles(pick)
                out[i] = (g, base, styles); used.update(p["id"] for p in pick); got += 1
            elif cn: later.append((i, c))
        for i, c in later:                               # by name alone, when it can only be one product
            cn = cn_of[i]
            if same[(cn, _holo(c.get("r")))] != 1: continue
            ps = [p for p in g["products"] if nname(p.get("n")) == cn] or [p for p in g["products"] if _fits(cn, nname(p.get("n")))]
            ps = [p for p in ps if p["id"] not in used]
            if len(ps) > 1 and len({nnum(p.get("no"))[0] for p in ps}) == 1:
                base, styles = split_styles(ps)           # one card listed plain and in special styles
                out[i] = (g, base, styles); used.update(p["id"] for p in ps); got += 1
                continue
            if len(ps) > 1:
                ps = [p for p in ps if _holo(p.get("r")) == _holo(c.get("r"))]
            if len(ps) == 1:
                out[i] = (g, ps[0]); used.add(ps[0]["id"]); got += 1
        stats.append((sid, cs[0][1].get("sn"), len(cs), g["name"] + f" [{how}]", got))
    return out, stats


def classic_reprints(cards, groups, lang="en"):
    """Classic Collection reprints (Celebrations 25th, 30th Celebration): old cards reprinted with an anniversary
    stamp, still carrying the ORIGINAL card's number ("Dark Gyarados 8/82") — but the main catalog numbers them
    CC001, CC002... so a scan of the number could only ever find the original, at the original's price.
    Gives each reprint the number printed on it (from TCGplayer's listing) and notes which card it reprints
    ("rp"), so the app can offer both. Returns how many were fixed."""
    sets = collections.defaultdict(list)
    for i, c in enumerate(cards):
        if c.get("l", "en") == lang and "classic collection" in (c.get("sn") or "").lower(): sets[c["s"]].append(i)
    if not sets: return 0
    originals = collections.defaultdict(list)
    for i, c in enumerate(cards):
        if c.get("l", "en") == lang and c["s"] not in sets: originals[nnum(c.get("no"))[0]].append(i)
    fixed = 0
    for g in groups:
        if "classic collection" not in (g.get("name") or "").lower(): continue
        names = collections.Counter(nname(p.get("n")) for p in g["products"])
        best = max(sets, key=lambda sid: sum(1 for i in sets[sid] if names.get(nname(cards[i].get("n")))), default=None)
        if best is None or sum(1 for i in sets[best] if names.get(nname(cards[i].get("n")))) < 0.5 * len(sets[best]): continue
        left = list(g["products"]); todo = list(sets[best])
        for exact in (True, False):
            for i in list(todo):
                cn = nname(cards[i].get("n"))
                p = next((p for p in left if (nname(p.get("n")) == cn if exact else (len(cn) >= 4 and _fits(cn, nname(p.get("n")))))), None)
                if p is None: continue
                left.remove(p); todo.remove(i)
                n, tot = nnum(p.get("no"))
                if not n: continue
                c = cards[i]
                c["no"] = n
                if tot.isdigit(): c["so"] = c["st"] = int(tot)
                # the card this is a reprint of: same number, same set size, same name, in an ordinary set
                orig = [j for j in originals.get(n, ()) if tot and str(cards[j].get("so") or "") == tot
                        and _fits(nname(cards[j].get("n")), nname(p.get("n")))]
                if len(orig) == 1: c["rp"] = cards[orig[0]]["i"]
                fixed += 1
    return fixed


SKIP_GROUP = ("world championship", "jumbo")


def cross_styles(cards, groups, lang, m):
    """Special versions that TCGplayer files away from the card's own set — Prize Pack stamps, League stamps,
    cosmos / cracked-ice holos from blisters and decks, holiday-calendar stamps. Each is paired with the catalog
    card of the same number, set size and name ("Volcanion - 25/114 (Pokemon League)" -> Steam Siege 25/114), or
    for promo-numbered ones by a number + name only one card has. Returns {card index: {style: (group, product)}}."""
    used = set()
    for hit in m.values():
        used.add(hit[1]["id"])
        for sp in (hit[2] if len(hit) > 2 else {}).values(): used.add(sp["id"])
    full = collections.defaultdict(list); loose = collections.defaultdict(list)
    for i, c in enumerate(cards):
        if c.get("l", "en") != lang: continue
        n = nnum(c.get("no"))[0]
        if not n: continue
        for tot in {str(c.get("so") or ""), str(c.get("st") or "")} - {"", "0"}:
            full[(n, tot)].append(i)
        loose[n].append(i)
    def name_of(c): return nname(c.get("n") if lang == "en" else (c.get("en") or ""))
    out = collections.defaultdict(dict)
    for g in groups:
        gname = g.get("name") or ""
        if any(w in gname.lower() for w in SKIP_GROUP): continue
        for p in g["products"]:
            if p["id"] in used or not p.get("n"): continue
            low = p["n"].lower()
            if "code card" in low or "jumbo" in low or "oversize" in low: continue
            st = style_of(p["n"], gname, strict=True)
            if not st: continue
            n, tot = nnum(p.get("no")); pn = nname(p["n"])
            if not n or not pn: continue
            if tot:
                cand = [i for i in full.get((n, tot), ()) if _fits(name_of(cards[i]), pn)]
            else:
                cand = [i for i in loose.get(n, ()) if name_of(cards[i]) == pn]
            if len(cand) != 1: continue                     # no such card, or can't tell which
            out[cand[0]].setdefault(st, (g, p))
    return out


def _add_styles(tp, src, styles):
    """styles: {style: (group, product)}. Adds each one's price to [tp] ("s:cosmos-holo"); a "=finish" entry
    fills that ordinary finish when the card's own listing has no price for it."""
    for st, (g, sp) in styles.items():
        rows = [x for x in g["prices"] if x["id"] == sp["id"] and (x.get("m") is not None or x.get("lo") is not None)]
        if not rows: continue
        x = max(rows, key=lambda r: (r.get("m") or 0, r.get("lo") or 0))
        key = st[1:] if st.startswith("=") else "s:" + st
        if key in tp: continue
        tp[key] = [x.get("m"), x.get("lo"), x.get("mid")]
        src[key] = (x["id"], x.get("t"))


def _price_rows(g, pid):
    tp = {}; src = {}
    for x in g["prices"]:
        if x["id"] != pid: continue
        k = SUB.get(x.get("t") or "")
        if k and (x.get("m") is not None or x.get("lo") is not None or x.get("mid") is not None):
            tp[k] = [x.get("m"), x.get("lo"), x.get("mid")]
            src[k] = (x["id"], x.get("t"))
    return tp, src


def missing_cards(cards, groups, lang, m, today, namer=None):
    """Cards TCGplayer lists that the main catalog doesn't have at all (starter sets, deck sets, promos, vending
    cards...). Returned as new catalog cards with TCGplayer's picture and price, so they can be recognised by
    their artwork. TCGplayer names them in English; the Japanese name is filled in from the Pokédex when the
    Pokémon can be told from the name."""
    used = {hit[1]["id"] for hit in m.values()}
    # which catalog set each TCGplayer set was paired with, and the numbers / names that set already has
    paired = collections.defaultdict(lambda: (set(), set()))
    by_set = collections.defaultdict(list)
    for c in cards:
        if c.get("l", "en") == lang: by_set[c["s"]].append(c)
    done = set()
    for i, hit in m.items():
        gid = hit[0]["id"]; sid_ = cards[i]["s"]
        if (gid, sid_) in done: continue
        done.add((gid, sid_))
        nums, names = paired[gid]
        for c in by_set[sid_]:
            nums.add(nnum(c.get("no"))[0]); names.add(nname(c.get("en") or c.get("n") or ""))
    species = []
    if namer is not None:
        species = sorted(((nname(n), sid) for sid, n in namer.en.items() if n), key=lambda x: -len(x[0]))
    have = {c["i"] for c in cards if c.get("l", "en") == lang}
    out = []
    for g in groups:
        nums, names = paired.get(g["id"], (set(), set()))
        title = g.get("name") or ""
        code = title.split(":")[0].strip() if ":" in title else ""
        if not re.fullmatch(r"[A-Za-z0-9.\-]{1,8}", code or ""): code = f"T{g['id']}"
        setname = title.split(":", 1)[1].strip() if ":" in title else title
        for p in g["products"]:
            if p["id"] in used or not p.get("img") or not p.get("n"): continue
            low = p["n"].lower()
            if "code card" in low or "jumbo" in low or "oversize" in low: continue
            n, tot = nnum(p.get("no"))
            pn = nname(p["n"])
            if (n and n in nums) or (pn and pn in names): continue        # the catalog has it; it just wasn't paired
            tp, src = _price_rows(g, p["id"])
            en = re.sub(r"\s+", " ", re.sub(r"\s*-\s*[\w]+/[\w]+\s*$", "", p["n"])).strip()
            sid = next((s_ for sn_, s_ in species if len(sn_) >= 3 and sn_ in pn), 0)
            ja = namer.ja.get(sid) if (namer is not None and sid) else None
            if ja and re.search(r"\bex\b", en): ja += "ex"
            number = (p.get("no") or "").split("/")[0].strip()
            cid = f"{code}-{number}" if number else f"{code}-tp{p['id']}"
            if cid in have: cid = f"{cid}-tp{p['id']}"
            have.add(cid)
            flags = "".join(f for f, k in (("N", "normal"), ("R", "reverse-holofoil"), ("H", "holofoil")) if k in tp)
            c = {"i": cid, "l": lang, "n": ja or en, "no": number, "s": code, "sn": setname,
                 "so": int(tot) if str(tot or "").isdigit() else 0, "st": int(tot) if str(tot or "").isdigit() else 0,
                 "r": "" if p.get("r") in (None, "None") else p["r"], "v": flags, "d": int(sid or 0), "img": p["img"], "en": en}
            if p.get("u"): c["u"] = p["u"]
            if tp:
                c["tp"] = tp; c["pu"] = today; c["_src"] = src
            out.append(c)
    return out


ADDED = {}
STYLED = {}
REPRINTS = {}


def apply(cards, groups, lang, today, namer=None):
    """Add TCGplayer pictures / prices / links to [cards] of one language. Returns (pictures, prices) added.
    For Japanese, cards TCGplayer has that the catalog lacks are appended to [cards] (count in ADDED[lang])."""
    if lang == "en": REPRINTS[lang] = classic_reprints(cards, groups, lang)
    m, stats = match_singles(cards, groups, lang) if lang == "en" else match_local(cards, groups, lang)
    if lang != "en":
        extra = missing_cards(cards, groups, lang, m, today, namer)
        ADDED[lang] = len(extra)
        cards.extend(extra)
    pics = prices = 0
    for i, hit in m.items():
        g, p = hit[0], hit[1]
        styles = hit[2] if len(hit) > 2 else {}
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
        # Special foil patterns and stamps of this card, each priced on its own ("s:master-ball").
        _add_styles(tp, src, {st: (g, sp) for st, sp in styles.items()})
        if tp:
            c["tp"] = tp; c["pu"] = today; prices += 1
            c["_src"] = src                         # build-time only; removed before the catalog is written
    # ...and the special versions filed under catch-all sets (Prize Pack, League, blister and deck exclusives).
    STYLED[lang] = 0
    for i, styles in cross_styles(cards, groups, lang, m).items():
        c = cards[i]
        tp = c.get("tp") or {}; src = c.get("_src") or {}
        before = len(tp)
        _add_styles(tp, src, styles)
        if len(tp) > before:
            c["tp"] = tp; c["_src"] = src; c.setdefault("pu", today); STYLED[lang] += 1
    return pics, prices, stats
