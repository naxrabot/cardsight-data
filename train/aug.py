"""Turns a clean catalog picture of a card into what a phone sees on a livestream: small, tilted, in a
sleeve with glare and marker writing, fingers on it, chat text over it, other cards behind it."""
import cv2, numpy as np

W, H = 192, 256          # what the picture model sees (card-shaped)
CW, CH = 240, 320        # working canvas

def _rand_color(rng): return tuple(int(x) for x in rng.integers(0, 256, 3))

def clean_view(card, rng=None, jitter=False):
    """The catalog picture as the model sees it (optionally very slightly varied)."""
    img = card
    if jitter and rng is not None:
        h, w = img.shape[:2]; m = rng.uniform(0, 0.03, 4)
        img = img[int(h * m[0]):h - int(h * m[1]), int(w * m[2]):w - int(w * m[3])]
        img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

def _glare(img, rng):
    h, w = img.shape[:2]; out = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.random()
        ang = rng.uniform(0, np.pi); d = (xx - rng.uniform(0, w)) * np.cos(ang) + (yy - rng.uniform(0, h)) * np.sin(ang)
        if kind < 0.45:      # band of light across the sleeve / foil
            a = np.exp(-(d / rng.uniform(0.06, 0.30) / w) ** 2)
        else:                # blob
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            a = np.exp(-(((xx - cx) / (w * rng.uniform(0.15, 0.6))) ** 2 + ((yy - cy) / (h * rng.uniform(0.1, 0.5))) ** 2))
        a = (a * rng.uniform(0.3, 0.95))[..., None]
        if rng.random() < 0.5:   # rainbow sheen (holo foil)
            hue = ((d / w * rng.uniform(0.8, 3.0) + rng.random()) % 1.0 * 179).astype(np.uint8)
            hsv = np.stack([hue, np.full_like(hue, int(rng.integers(90, 220))), np.full_like(hue, 255)], -1)
            col = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        else:
            col = np.full((h, w, 3), 255.0, np.float32) * np.array([rng.uniform(.9, 1), rng.uniform(.9, 1), rng.uniform(.9, 1)], np.float32)
        out = out * (1 - a) + col * a
    return np.clip(out, 0, 255).astype(np.uint8)

def _sparkle(img, rng):
    """Confetti / cracked-ice / cosmos foil: lots of small bright coloured flecks."""
    h, w = img.shape[:2]; out = img.astype(np.float32)
    n = int(rng.integers(60, 500)); layer = np.zeros((h, w, 3), np.float32)
    xs = rng.integers(0, w, n); ys = rng.integers(int(h * rng.uniform(0, 0.12)), int(h * rng.uniform(0.5, 1.0)), n)
    for x, y in zip(xs, ys):
        hsv = np.uint8([[[int(rng.integers(0, 180)), int(rng.integers(0, 160)), 255]]])
        col = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        r = int(rng.integers(1, 5))
        if rng.random() < 0.5: cv2.circle(layer, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
        else: cv2.rectangle(layer, (int(x) - r, int(y) - r), (int(x) + r, int(y) + r), col, -1)
    a = (layer.max(-1, keepdims=True) / 255.0) * rng.uniform(0.35, 0.9)
    return np.clip(out * (1 - a) + layer * a, 0, 255).astype(np.uint8)

def _marker(img, rng):
    h, w = img.shape[:2]; img = img.copy()
    col = [(200, 20, 30), (20, 20, 20), (30, 40, 180), (230, 230, 230)][int(rng.integers(0, 4))]
    th = int(rng.integers(max(2, w // 60), max(3, w // 22)))
    if rng.random() < 0.6:   # a price written on the sleeve
        txt = ["$", ""][int(rng.integers(0, 2))] + str(int(rng.integers(1, 999)))
        cv2.putText(img, txt, (int(rng.uniform(0, w * .5)), int(rng.uniform(h * .12, h * .95))),
                    [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_ITALIC][int(rng.integers(0, 3))],
                    rng.uniform(w / 220, w / 70), col, th, cv2.LINE_AA)
    else:                    # a scribble
        pts = np.cumsum(rng.normal(0, w * 0.12, (int(rng.integers(4, 9)), 2)), 0) + [rng.uniform(.2, .8) * w, rng.uniform(.2, .8) * h]
        cv2.polylines(img, [pts.astype(np.int32)], False, col, th, cv2.LINE_AA)
    if rng.random() < 0.5:   # big looping handwriting across the lower half (prices, initials)
        x = rng.uniform(0.1, 0.4) * w; y = rng.uniform(0.5, 0.85) * h
        for _ in range(int(rng.integers(2, 5))):
            rx, ry = w * rng.uniform(0.04, 0.12), h * rng.uniform(0.05, 0.14)
            cv2.ellipse(img, (int(x), int(y)), (int(rx), int(ry)), rng.uniform(-30, 30), rng.uniform(0, 90), rng.uniform(200, 360), col, th, cv2.LINE_AA)
            x += rx * rng.uniform(1.4, 2.4); y += rng.normal(0, h * 0.02)
    return img

def _color(img, rng):
    x = img.astype(np.float32) / 255
    x = x ** rng.uniform(0.7, 1.4)
    x = (x - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.12, 0.12)
    x = x * (1 + rng.uniform(-0.12, 0.12, 3)).astype(np.float32)                 # colour cast
    g = x.mean(-1, keepdims=True); x = g + (x - g) * rng.uniform(0.6, 1.5)       # saturation
    return np.clip(x * 255, 0, 255).astype(np.uint8)

def _background(rng, others):
    k = rng.random()
    if k < 0.25 or not others:
        c1, c2 = np.array(_rand_color(rng), np.float32), np.array(_rand_color(rng), np.float32)
        t = np.linspace(0, 1, CH, dtype=np.float32)[:, None, None]
        bg = np.broadcast_to(c1 * (1 - t) + c2 * t, (CH, CW, 3)).astype(np.uint8).copy()
    elif k < 0.45:
        o = others[int(rng.integers(0, len(others)))]
        bg = cv2.GaussianBlur(cv2.resize(o, (CW, CH)), (0, 0), rng.uniform(3, 12))
    else:                     # a rack / wall of other cards
        bg = np.full((CH, CW, 3), _rand_color(rng), np.uint8)
        cw = int(rng.uniform(40, 110)); ch = int(cw * 1.4); ox, oy = int(rng.integers(0, cw)), int(rng.integers(0, ch))
        gap = int(rng.integers(0, 10))
        for y in range(-oy, CH, ch + gap):
            for x in range(-ox, CW, cw + gap):
                o = cv2.resize(others[int(rng.integers(0, len(others)))], (cw, ch), interpolation=cv2.INTER_AREA)
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + cw, CW), min(y + ch, CH)
                if x1 > x0 and y1 > y0: bg[y0:y1, x0:x1] = o[y0 - y:y1 - y, x0 - x:x1 - x]
        if rng.random() < 0.6: bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 3))
    return np.clip(bg.astype(np.float32) * rng.uniform(0.45, 1.05), 0, 255).astype(np.uint8)

_WORDS = "joined lol how much is that nice pull bid fire shipping pikachu wow love it can you show yes pls sold won grail mint packs let's go the next one".split()

def _overlays(img, rng):
    h, w = img.shape[:2]
    if rng.random() < 0.45:                                   # fingers holding the card
        for _ in range(int(rng.integers(1, 4))):
            skin = (int(rng.integers(150, 250)), int(rng.integers(110, 200)), int(rng.integers(90, 180)))
            side = rng.random()
            c = (int(rng.uniform(0, w)), int(h * rng.uniform(0.8, 1.05))) if side < 0.5 else \
                (int(w * (rng.uniform(-0.05, 0.12) if side < 0.75 else rng.uniform(0.88, 1.05))), int(rng.uniform(0.2, 1) * h))
            cv2.ellipse(img, c, (int(w * rng.uniform(0.06, 0.22)), int(h * rng.uniform(0.08, 0.34))), rng.uniform(0, 180), 0, 360, skin, -1, cv2.LINE_AA)
    if rng.random() < 0.15:                                   # "… won the auction!" banner / giveaway pop-up
        c = (int(w * rng.uniform(0.35, 0.65)), int(h * rng.uniform(0.15, 0.5))); r = int(w * rng.uniform(0.08, 0.16))
        cv2.circle(img, c, r, [(250, 225, 40), (20, 20, 20), (255, 255, 255)][int(rng.integers(0, 3))], -1, cv2.LINE_AA)
        for k, t in enumerate(("user%d" % rng.integers(1, 99), "Won the auction!")):
            fs = rng.uniform(0.5, 0.8) - 0.15 * k
            cv2.putText(img, t, (c[0] - int(w * 0.3), c[1] + r + 22 + k * 22), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        if rng.random() < 0.5:                                # confetti
            for _ in range(int(rng.integers(20, 80))):
                p0 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
                cv2.rectangle(img, p0, (p0[0] + int(rng.integers(2, 7)), p0[1] + int(rng.integers(2, 7))), _rand_color(rng), -1)
    if rng.random() < 0.55:                                   # chat scrolling over the picture
        y = int(rng.uniform(0.15, 0.8) * h); x = int(rng.uniform(-0.3, 0.3) * w)
        for _ in range(int(rng.integers(1, 6))):
            if rng.random() < 0.5: cv2.circle(img, (max(6, x - 10), y - 5), int(rng.integers(5, 10)), _rand_color(rng), -1, cv2.LINE_AA)
            s = " ".join(_WORDS[int(i)] for i in rng.integers(0, len(_WORDS), int(rng.integers(1, 6))))
            fs = rng.uniform(0.35, 0.7); col = (255, 255, 255) if rng.random() < 0.8 else (200, 200, 200)
            cv2.putText(img, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, int(rng.integers(1, 3)), cv2.LINE_AA)
            y += int(fs * 42)
    if rng.random() < 0.3:                                    # the scanner's own dashed box
        m = rng.uniform(0.01, 0.12, 4); x0, y0, x1, y1 = int(w * m[0]), int(h * m[1]), int(w * (1 - m[2])), int(h * (1 - m[3]))
        for a in range(x0, x1, 14):
            cv2.line(img, (a, y0), (min(a + 8, x1), y0), (245, 205, 60), 2); cv2.line(img, (a, y1), (min(a + 8, x1), y1), (245, 205, 60), 2)
        for a in range(y0, y1, 14):
            cv2.line(img, (x0, a), (x0, min(a + 8, y1)), (245, 205, 60), 2); cv2.line(img, (x1, a), (x1, min(a + 8, y1)), (245, 205, 60), 2)
    if rng.random() < 0.2:                                    # price bar / listing panel covering the bottom
        y = int(h * rng.uniform(0.68, 0.92)); a = rng.uniform(0.6, 1.0)
        img[y:] = (img[y:].astype(np.float32) * (1 - a) + np.array([18, 22, 34], np.float32) * a).astype(np.uint8)
    return img

def stream_view(card, rng, others=()):
    """One simulated livestream view of [card] (RGB uint8)."""
    c = cv2.resize(card, (200, 280), interpolation=cv2.INTER_AREA)
    c = _color(c, rng)
    if rng.random() < 0.25: c = _sparkle(c, rng)
    if rng.random() < 0.65: c = _glare(c, rng)
    if rng.random() < 0.4: c = _marker(c, rng)
    if rng.random() < 0.5:                                    # sleeve / top-loader / slab edge around the card
        b = int(rng.integers(2, 14)); col = _rand_color(rng) if rng.random() < 0.4 else tuple(int(rng.integers(170, 255)) for _ in range(3))
        c = cv2.copyMakeBorder(c, b + int(rng.integers(0, 16)), b, b, b, cv2.BORDER_CONSTANT, value=col)
    h, w = c.shape[:2]
    bg = _background(rng, others)
    # where the card sits: mostly filling the view, a bit off-centre, tilted and seen at an angle
    ch = CH * rng.uniform(0.58, 1.22); cw = ch * w / h * rng.uniform(0.85, 1.12)
    cx, cy = CW / 2 + rng.normal(0, CW * 0.07), CH / 2 + rng.normal(0, CH * 0.07)
    ang = np.deg2rad(rng.normal(0, 7)); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    corners = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]]) + rng.normal(0, 0.035, (4, 2)) * [cw, ch]
    dst = (corners @ R.T + [cx, cy]).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    warped = cv2.warpPerspective(c, M, (CW, CH), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (CW, CH), flags=cv2.INTER_LINEAR)[..., None].astype(np.float32) / 255
    img = (warped * mask + bg * (1 - mask)).astype(np.uint8)
    img = _overlays(img, rng)
    # video quality: the card is small on screen, soft and blocky
    s = rng.uniform(0.22, 1.0)
    if s < 0.95:
        img = cv2.resize(cv2.resize(img, (max(24, int(CW * s)), max(32, int(CH * s))), interpolation=cv2.INTER_AREA), (CW, CH),
                         interpolation=cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_CUBIC)
    if rng.random() < 0.4: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.6))
    if rng.random() < 0.8:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(18, 85))])
        if ok: img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
