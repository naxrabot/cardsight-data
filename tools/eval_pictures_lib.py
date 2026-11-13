import io, random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

def stream_view(card, card_px):
    w, h = card.size
    bg = Image.new("RGB", (int(w * 1.5), int(h * 1.35)), tuple(random.randint(20, 120) for _ in range(3)))
    d = ImageDraw.Draw(bg)
    for _ in range(6):
        x, y = random.randint(0, bg.width), random.randint(0, bg.height)
        d.ellipse([x, y, x + 120, y + 120], fill=tuple(random.randint(60, 200) for _ in range(3)))
    rc = card.rotate(random.uniform(-6, 6), expand=True, fillcolor=(0, 0, 0))
    bg.paste(rc, ((bg.width - rc.width) // 2 + random.randint(-10, 10), (bg.height - rc.height) // 2 + random.randint(-10, 10)))
    g = Image.new("L", bg.size, 0); gd = ImageDraw.Draw(g)
    gx = random.randint(0, bg.width)
    gd.ellipse([gx - 80, -40, gx + 80, bg.height // 2], fill=random.randint(30, 90))
    bg = Image.composite(Image.new("RGB", bg.size, (255, 255, 255)), bg, g.filter(ImageFilter.GaussianBlur(40)))
    bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.7, 1.15))
    s = card_px / w
    small = bg.resize((int(bg.width * s), int(bg.height * s)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.5))
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=40)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    cw, ch = out.width / 1.5, out.height / 1.35
    m = random.uniform(0.0, 0.05)
    l, t = (out.width - cw) / 2 - cw * m, (out.height - ch) / 2 - ch * m
    return out.crop((int(l), int(t), int(l + cw * (1 + 2 * m)), int(t + ch * (1 + 2 * m))))

