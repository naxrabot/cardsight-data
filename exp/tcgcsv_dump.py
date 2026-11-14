"""One-off: save TCGplayer's list of single cards (via tcgcsv.com) so matching them to the catalog can be
worked out offline. Keeps only the fields needed."""
import concurrent.futures as cf, gzip, json, os, sys
sys.path.insert(0, ".")
from build_data import http, TCGCSV, log
os.makedirs("out", exist_ok=True)
dump = {}
for cat in (3, 85):
    groups = json.loads(http(f"{TCGCSV}/{cat}/groups"))["results"]
    def one(g):
        gid = g["groupId"]
        try:
            prods = json.loads(http(f"{TCGCSV}/{cat}/{gid}/products", tries=4) or b'{"results":[]}')["results"]
            prices = json.loads(http(f"{TCGCSV}/{cat}/{gid}/prices", tries=4) or b'{"results":[]}')["results"]
        except Exception:
            return g, [], []
        ps = []
        for p in prods:
            ext = {e.get("name"): e.get("value") for e in (p.get("extendedData") or [])}
            if "Number" not in ext and "Rarity" not in ext and "HP" not in ext: continue
            ps.append({"id": p["productId"], "n": p.get("name"), "clean": p.get("cleanName"), "img": p.get("imageUrl"), "u": p.get("url"),
                       "no": ext.get("Number"), "r": ext.get("Rarity"), "hp": ext.get("HP")})
        pr = [{"id": x["productId"], "t": x.get("subTypeName"), "m": x.get("marketPrice"), "lo": x.get("lowPrice"), "mid": x.get("midPrice"), "hi": x.get("highPrice")} for x in prices]
        return g, ps, pr
    out = []
    with cf.ThreadPoolExecutor(8) as ex:
        for g, ps, pr in ex.map(one, groups):
            out.append({"id": g["groupId"], "name": g.get("name"), "abbr": g.get("abbreviation"), "date": (g.get("publishedOn") or "")[:10], "products": ps, "prices": pr})
    dump[str(cat)] = out
    log(cat, len(out), "groups", sum(len(g["products"]) for g in out), "single cards")
json.dump(dump, gzip.open("out/tcgcsv_dump.json.gz", "wt"))
