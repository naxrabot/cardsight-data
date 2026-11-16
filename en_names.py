"""
English names for Japanese and Chinese cards.

Uses PokéAPI's species-name table (English, Japanese, Traditional and Simplified Chinese) plus the
card-name conventions (ex / V / VMAX / GX, Mega, regional forms, "Team Rocket's", Delta Species...).
Trainer and Energy cards are translated from a small table of common names; others stay untranslated.
"""
import csv
import io
import re
import unicodedata
import urllib.request

CSV_URL = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/pokemon_species_names.csv"
EN, JA_KANA, JA, ZH_HANT, ZH_HANS = 9, 1, 11, 4, 12

# Prefixes in the local name -> English prefix. Checked in order (longest first).
PREFIXES = [
    ("ロケット団の", "Team Rocket's "), ("ロケットだんの", "Team Rocket's "),
    ("火箭隊的", "Team Rocket's "), ("火箭队的", "Team Rocket's "),
    ("ガラル", "Galarian "), ("伽勒爾", "Galarian "), ("伽勒尔", "Galarian "),
    ("アローラ", "Alolan "), ("阿羅拉", "Alolan "), ("阿罗拉", "Alolan "),
    ("ヒスイ", "Hisuian "), ("洗翠", "Hisuian "),
    ("パルデア", "Paldean "), ("帕底亞", "Paldean "), ("帕底亚", "Paldean "),
    ("メガ", "Mega "), ("超級", "Mega "), ("超级", "Mega "),
    ("ダーク", "Dark "), ("わるい", "Dark "), ("ライト", "Light "),
    ("ひかる", "Shining "), ("かがやく", "Radiant "), ("光輝", "Radiant "), ("光辉", "Radiant "),
    ("Nの", "N's "), ("マリィの", "Marnie's "), ("エリカの", "Erika's "), ("カスミの", "Misty's "),
    ("タケシの", "Brock's "), ("ナツメの", "Sabrina's "), ("カツラの", "Blaine's "),
    ("マチスの", "Lt. Surge's "), ("キョウの", "Koga's "), ("サカキの", "Giovanni's "),
    ("ホップの", "Hop's "), ("リーリエの", "Lillie's "), ("ナンジャモの", "Iono's "),
    ("シロナの", "Cynthia's "), ("スグリの", "Kieran's "), ("ゼイユの", "Carmine's "),
    ("イーサンの", "Ethan's "), ("アカギの", "Cyrus's "), ("ボタンの", "Arven's "),
]

# Common trainer / energy cards (Japanese & Chinese names -> English).
TRAINERS = {
    "ボールのため": "", "博士の研究": "Professor's Research", "ナンジャモ": "Iono", "ボスの指令": "Boss's Orders",
    "ネストボール": "Nest Ball", "ハイパーボール": "Ultra Ball", "モンスターボール": "Poké Ball",
    "クイックボール": "Quick Ball", "ふしぎなアメ": "Rare Candy", "ポケモンいれかえ": "Switch",
    "スーパーエネルギー回収": "Superior Energy Retrieval", "ペパー": "Arven", "ネモ": "Nemona",
    "ジャッジマン": "Judge", "シロナ": "Cynthia", "マリィ": "Marnie", "リーリエ": "Lillie",
    "エリカのおもてなし": "Erika's Invitation", "ミモザ": "Miriam", "ボタン": "Penny",
    "基本草エネルギー": "Basic Grass Energy", "基本炎エネルギー": "Basic Fire Energy",
    "基本水エネルギー": "Basic Water Energy", "基本雷エネルギー": "Basic Lightning Energy",
    "基本超エネルギー": "Basic Psychic Energy", "基本闘エネルギー": "Basic Fighting Energy",
    "基本悪エネルギー": "Basic Darkness Energy", "基本鋼エネルギー": "Basic Metal Energy",
    "基本草能量": "Basic Grass Energy", "基本火能量": "Basic Fire Energy", "基本水能量": "Basic Water Energy",
    "基本雷能量": "Basic Lightning Energy", "基本超能量": "Basic Psychic Energy",
    "基本鬥能量": "Basic Fighting Energy", "基本恶能量": "Basic Darkness Energy",
    "基本惡能量": "Basic Darkness Energy", "基本鋼能量": "Basic Metal Energy", "基本钢能量": "Basic Metal Energy",
    "博士的研究": "Professor's Research", "老大的指令": "Boss's Orders", "高級球": "Ultra Ball",
    "巢穴球": "Nest Ball", "神奇糖果": "Rare Candy", "寶可夢交替": "Switch", "宝可梦交替": "Switch",
    "奇樹": "Iono", "奇树": "Iono",
    "寶可夢捕捉器": "Pokémon Catcher", "ポケモンキャッチャー": "Pokémon Catcher", "裁判": "Judge",
    "超級球": "Great Ball", "スーパーボール": "Great Ball", "精靈球": "Poké Ball", "精灵球": "Poké Ball",
    "霧之水晶": "Fog Crystal", "きりのクリスタル": "Fog Crystal", "瑪俐": "Marnie", "瑪俐的驕傲": "Marnie's Pride",
    "マリィのプライド": "Marnie's Pride", "先機球": "Quick Ball", "先机球": "Quick Ball", "妮莫": "Nemona",
    "ポケギア3.0": "Pokégear 3.0", "寶可夢通3.0": "Pokégear 3.0", "なかよしポフィン": "Buddy-Buddy Poffin",
    "改造ハンマー": "Enhanced Hammer", "クラッシュハンマー": "Crushing Hammer", "ダブル無色エネルギー": "Double Colorless Energy",
    "等級球": "Level Ball", "レベルボール": "Level Ball", "カウンターゲイン": "Counter Gain", "アカマツ": "Crispin",
    "こだわりハチマキ": "Choice Belt", "あなぬけのヒモ": "Escape Rope", "能量轉移": "Energy Switch",
    "エネルギーつけかえ": "Energy Switch", "エネルギー回収": "Energy Retrieval", "能量回收": "Energy Retrieval",
    "腐朽的盾": "Rusted Shield", "くちた盾": "Rusted Shield", "くちた剣": "Rusted Sword", "腐朽的劍": "Rusted Sword",
    "クラベル": "Clavell", "ふつうのつりざお": "Super Rod", "すごいつりざお": "Super Rod", "厲害釣竿": "Super Rod",
    "大地の器": "Earthen Vessel", "大地容器": "Earthen Vessel", "ポケモン回収サイクロン": "Pokémon Retrieval Cyclone",
    "夜のタンカ": "Night Stretcher", "夜間擔架": "Night Stretcher", "プライムキャッチャー": "Prime Catcher",
    "頂尖捕捉器": "Prime Catcher", "ロストスイーパー": "Lost Vacuum", "ふうせん": "Air Balloon", "勇気のおまもり": "Bravery Charm",
    "まけんきハチマキ": "Defiance Band", "ヒーローマント": "Hero's Cape", "森の封印石": "Forest Seal Stone",
    "博士の研究": "Professor's Research", "ボスの指令": "Boss's Orders", "老大的指令": "Boss's Orders",
    "博士的研究": "Professor's Research", "ペパー": "Arven", "派帕": "Arven", "ネモ": "Nemona",
    "ミモザ": "Miriam", "キハダ": "Dendra", "リップ": "Raihan", "ツツジ": "Roxanne",
    "サポーター": "Supporter", "ダブルターボエネルギー": "Double Turbo Energy", "ジェットエネルギー": "Jet Energy",
    "ギフトエネルギー": "Gift Energy", "レガシーエネルギー": "Legacy Energy", "リバーサルエネルギー": "Reversal Energy",
}

PAREN_RE = re.compile(r"[（(][^）)]*[）)]")

# Card-type endings written in Latin letters (kept as printed, e.g. "ex" vs "EX").
SUFFIX_RE = re.compile(r"\s*(VMAX|VSTAR|V-UNION|V|GX|EX|Ex|ex|BREAK|LV\.X|LEGEND|Prime|◇|☆)\s*$")


def _nfkc(s):
    return unicodedata.normalize("NFKC", s or "").strip()


def load_species():
    raw = urllib.request.urlopen(CSV_URL, timeout=60).read().decode("utf-8")
    en, local = {}, {}
    for row in csv.DictReader(io.StringIO(raw)):
        sid, lang, name = int(row["pokemon_species_id"]), int(row["local_language_id"]), row["name"]
        if lang == EN:
            en[sid] = name
        elif lang in (JA_KANA, JA, ZH_HANT, ZH_HANS) and name:
            local[_nfkc(name)] = sid
    # longest local names first so "ヒトカゲ" doesn't beat "メガヒトカゲ"-style overlaps
    names = sorted(local, key=len, reverse=True)
    return en, local, names


class Namer:
    def __init__(self):
        self.en, self.local, self.names = load_species()
        # Official Japanese name by Pokédex number (kana), for cards whose listed name is unusable.
        self.ja = {sid: name for name, sid in self.local.items() if re.fullmatch(r"[\u30A0-\u30FF\u3040-\u309F♀♂:・ー2Z]+", name)}

    def proper_japanese(self, name, dex):
        """Some old Japanese sets are listed with machine-translated names ("政治" for Politoed, "beedrill").
        Returns the official name when the listed one doesn't even contain it, else None."""
        official = self.ja.get(dex)
        n = _nfkc(name)
        if not official or official in n:
            return None
        # keep what was right about the listed name: the card type ("ex") and the Delta Species mark
        kind = SUFFIX_RE.search(PAREN_RE.sub("", n).strip())
        delta = "デルタ種" in n or "δ" in n
        return official + (kind.group(1) if kind else "") + ("(デルタ種)" if delta else "")

    def english(self, name, dex):
        n = _nfkc(name)
        if not n:
            return None
        if n in TRAINERS:
            return TRAINERS[n] or None
        # "博士的研究（山梨博士）" -> "博士的研究" (same card, different professor art)
        bare = PAREN_RE.sub("", n).strip()
        if bare in TRAINERS and "デルタ" not in n:
            return TRAINERS[bare] or None

        # Strip decorations we translate separately.
        delta = False
        star = False
        for d in ("（デルタ種）", "(デルタ種)", "δ", "（δ種）"):
            if d in n:
                delta = True
                n = n.replace(d, "")
        if n.endswith("の星") or n.endswith("☆"):
            star = True
            n = n.removesuffix("の星").removesuffix("☆")
        # "<派帕的>藏飽栗鼠": drop an untranslated owner in brackets (known owners are handled below)
        mo = re.match(r"^[<〈＜]([^>〉＞]+)[>〉＞]\s*(.+)$", n)
        if mo and not any(mo.group(1) + ">" in p[0] or p[0].startswith(mo.group(1)) for p in PREFIXES):
            n = mo.group(2)
        n = n.strip("<>〈〉＜＞ ")

        prefix = ""
        moved = True
        while moved:  # e.g. "ロケット団の" + "ダーク"
            moved = False
            for p, e in PREFIXES:
                if n.startswith(p) and len(n) > len(p):
                    prefix += e
                    n = n[len(p):].lstrip("<>〈〉＜＞ ")
                    moved = True
                    break

        suffix = ""
        m = SUFFIX_RE.search(n)
        if m:
            suffix = " " + m.group(1)
            n = n[: m.start()].strip()

        form = ""
        if n[-1:] in ("X", "Y") and n[:-1] in self.local:  # Mega Charizard X / Y
            form = " " + n[-1]
            n = n[:-1]

        sid = self.local.get(n)
        if sid is None and dex:
            sid = dex
        if sid is None:
            for loc in self.names:  # contained species name, e.g. form names
                if len(loc) >= 2 and loc in n:
                    sid = self.local[loc]
                    break
        species = self.en.get(sid) if sid else None
        if not species:
            return None
        out = prefix + species + form + suffix
        if delta:
            out += " δ"
        if star:
            out += " ☆"
        return out
