"""One-off look at what TCGplayer lists for new and upcoming Pokemon sets."""
import json, urllib.request, datetime
def get(u):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "cardsight-data"}), timeout=60).read())["results"]
today = datetime.date.today().isoformat()
for cat in (3, 85):
    gs = get(f"https://tcgcsv.com/tcgplayer/{cat}/groups")
    print("== category", cat, len(gs), "groups; keys", sorted(gs[0].keys()))
    for g in sorted(gs, key=lambda g: g["groupId"])[-14:]:
        ps = get(f"https://tcgcsv.com/tcgplayer/{cat}/{g['groupId']}/products")
        pr = get(f"https://tcgcsv.com/tcgplayer/{cat}/{g['groupId']}/prices")
        priced = {p["productId"] for p in pr if p.get("marketPrice") or p.get("lowPrice")}
        pre = [p for p in ps if (p.get("presaleInfo") or {}).get("isPresale")]
        dated = [p for p in ps if (p.get("presaleInfo") or {}).get("releasedOn")]
        print(g["groupId"], g["name"], "| pub", g.get("publishedOn"), "mod", g.get("modifiedOn"), "| products", len(ps), "priced", len(priced), "presale", len(pre), "with date", len(dated))
        for p in (pre or dated or ps)[:3]:
            print("     ", p["name"], "|", json.dumps(p.get("presaleInfo")), "| priced" if p["productId"] in priced else "| no price",
                  "| ext", [e.get("name") for e in (p.get("extendedData") or [])][:4])
    fut = [g for g in gs if (g.get("publishedOn") or "")[:10] > today]
    print("future-dated groups:", [(g["groupId"], g["name"], g["publishedOn"][:10]) for g in fut])
