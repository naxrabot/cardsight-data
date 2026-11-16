"""Check what tcgcsv's daily price archives look like (for price history / trends)."""
import sys, os, subprocess, json, datetime, urllib.request, time
sys.path.insert(0, ".")
from build_data import http
day = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
url = f"https://tcgcsv.com/archive/tcgplayer/prices-{day}.ppmd.7z"
t = time.time(); raw = http(url, tries=3, timeout=120); print("downloaded", len(raw or b""), "bytes in", round(time.time() - t, 1), "s", url, flush=True)
open("/tmp/a.7z", "wb").write(raw)
t = time.time(); r = subprocess.run(["7z", "x", "-y", "-o/tmp/arch", "/tmp/a.7z", f"{day}/3/*", f"{day}/85/*"], capture_output=True, text=True)
print("7z rc", r.returncode, round(time.time() - t, 1), "s", r.stdout[-600:], r.stderr[-300:], flush=True)
n = 0
for root, dirs, files in os.walk("/tmp/arch"):
    for f in files:
        n += 1
        if n <= 3:
            p = os.path.join(root, f); print(p, os.path.getsize(p)); print(open(p).read()[:500])
print("files", n)
print(subprocess.run("ls /tmp/arch | head; ls /tmp/arch/* | head", shell=True, capture_output=True, text=True).stdout)
