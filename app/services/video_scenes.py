"""Frame-by-frame scenes for the news video: article images, Ken Burns zoom, slide/fade-in
text and word-by-word captions. Each Scene draws a 1080x1920 frame for any time t."""
import io
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from app.core.config import settings
from app.crawlers.article_crawler import headers as CRAWL_HEADERS
from app.crawlers.url_guard import guarded_client

logger = logging.getLogger(__name__)

W, H = 1080, 1920
PAD = 90
# Design tokens from the frontend ("Tín hiệu")
PAPER, RULE = "#f2f5f8", "#d8e1ea"
INK, INK2, MUTED, FAINT = "#16202b", "#3e4955", "#5b6570", "#a3adb8"
ACCENT, ACCENT_400, ACCENT_950 = "#2650c8", "#5f80d9", "#0f1c42"
SIGNAL, SIGNAL_300, OCHRE, OCHRE_300, WHITE = "#0e7459", "#67b597", "#a8741a", "#dfb05a", "#ffffff"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]

# TikTok's UI covers roughly the bottom 380px and the right 120px, so text stays above ~1540.
ZOOM = 1.12  # Ken Burns on the sharp image: starts 1.12x and eases out to 1.0x


# ---------- helpers ----------

def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


@lru_cache(maxsize=32)
def font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    # VIDEO_FONT_PATH (one file for regular and bold) is for machines without a known Vietnamese font
    override = settings.VIDEO_FONT_PATH
    for p in ([override] if override else []) + (BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    logger.warning("No Vietnamese-capable TTF found; set VIDEO_FONT_PATH. Diacritics may break.")
    return ImageFont.load_default(size)


_measure = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def wrap(text: str, fnt, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if _measure.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def layout_words(text: str, fnt, max_w: int) -> List[Tuple[str, int, int]]:
    """(word, x, line_index) for each word, wrapped to max_w."""
    space = _measure.textlength(" ", font=fnt)
    out, x, line = [], 0.0, 0
    for w in text.split():
        ww = _measure.textlength(w, font=fnt)
        if x > 0 and x + ww > max_w:
            line, x = line + 1, 0.0
        out.append((w, int(x), line))
        x += ww + space
    return out


def word_starts(words: List[str]) -> List[float]:
    """Fraction of the speech at which each word starts, proportional to character count."""
    total = sum(len(w) for w in words) + max(len(words) - 1, 0)
    starts, acc = [], 0
    for w in words:
        starts.append(acc / total if total else 0.0)
        acc += len(w) + 1
    return starts


def tier_color(score: float) -> str:
    """Score accent on dark backgrounds (high = signal, mid = ochre)."""
    return SIGNAL_300 if score >= 8 else OCHRE_300 if score >= 6 else FAINT


def pulse_y(u: float, amp: float) -> float:
    """ECG-style trace for u in 0..1: small noise, one spike at 62% of the width. Returns -1..1."""
    noise = 0.08 * math.sin(u * 38) + 0.05 * math.sin(u * 91 + 1)
    spike = math.exp(-((u - 0.62) / 0.018) ** 2) * amp - 0.45 * math.exp(-((u - 0.67) / 0.02) ** 2) * amp
    return noise + spike


# ---------- article image ----------

async def fetch_article_image(url: str) -> Optional[Image.Image]:
    """og:image of an article page, or None. Uses the crawler's SSRF-guarded, size-capped client."""
    if not url:
        return None
    try:
        async with guarded_client(timeout=10.0, follow_redirects=True, headers=CRAWL_HEADERS) as client:
            page = await client.get(url)
            page.raise_for_status()
            soup = BeautifulSoup(page.text, "html.parser")
            meta = soup.find("meta", attrs={"property": "og:image"}) or soup.find(
                "meta", attrs={"name": "twitter:image"}
            )
            src = (meta.get("content") or "").strip() if meta else ""
            if not src:
                return None
            resp = await client.get(urljoin(str(page.url), src))
            resp.raise_for_status()
            if not resp.headers.get("content-type", "").startswith("image/"):
                return None
            img = Image.open(io.BytesIO(resp.content))
            img.load()
            if img.width < 480 or img.height < 270:
                return None
            return img.convert("RGB")
    except Exception as err:
        logger.info(f"No article image for {url}: {type(err).__name__}: {err}")
        return None


# ---------- scenes ----------

def _backdrop(image: Optional[Image.Image], dark: str) -> Image.Image:
    """Full-frame background: the article image blurred and darkened, or a flat dark color."""
    if image is None:
        return Image.new("RGB", (W, H), dark)
    small = ImageOps.fit(image, (W // 6, H // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
    bg = small.resize((W, H), Image.BILINEAR)
    return Image.blend(bg, Image.new("RGB", (W, H), dark), 0.72)


class Scene:
    """One segment of the video. frame(t) returns the RGB frame at second t of the scene.
    kinds: intro (hook over the top story's image), item (ranked story), outro (call to action)."""

    HERO_Y, HERO_H = 400, 620   # sharp image band
    TITLE_Y = 1060

    def __init__(self, seg: Dict[str, Any], index: int, count: int, speech: float, duration: float):
        self.seg, self.index, self.count = seg, index, count
        self.speech, self.duration = speech, duration
        self.kind = seg["kind"]
        self.global_start, self.global_total = 0.0, 1.0
        image = seg.get("image")
        self.bg_color = ACCENT_950 if self.kind == "outro" else "#0b111a"
        self.base = _backdrop(image if self.kind != "outro" else None, self.bg_color)
        d = ImageDraw.Draw(self.base)
        d.text((PAD, 70), "TECHPULSE", font=font(True, 38), fill=WHITE)
        if self.kind == "item" and seg.get("rank"):
            d.text((W - PAD - 110, 74), f"Top {seg['rank']}", font=font(True, 38), fill=SIGNAL_300, anchor="ra")
        credit = "Tóm tắt tự động bằng AI"
        if image is not None and seg.get("meta") and self.kind == "item":
            credit += f"  ·  Ảnh: {seg['meta']}"
        d.text((PAD, H - 120), credit, font=font(False, 30), fill="#8d98a5")

        self.hero = None
        if image is not None and self.kind == "item":
            self.hero = ImageOps.fit(image, (math.ceil(W * ZOOM), math.ceil(self.HERO_H * ZOOM)), Image.LANCZOS)

        # Caption: big, centered column, karaoke highlight
        size = {"item": 54, "intro": 84, "outro": 70}[self.kind]
        self.cap_font = font(True, size)
        self.cap_words = layout_words(seg["text"], self.cap_font, W - 2 * PAD)
        if self.kind == "item" and max((l for _, _, l in self.cap_words), default=0) > 4:
            self.cap_font = font(True, 46)
            self.cap_words = layout_words(seg["text"], self.cap_font, W - 2 * PAD)
        self.cap_starts = word_starts([w for w, _, _ in self.cap_words])
        self.cap_line_h = int(self.cap_font.size * 1.4)
        lines = max((l for _, _, l in self.cap_words), default=0) + 1
        self.cap_h = lines * self.cap_line_h

        if self.kind == "item":
            tfont = font(True, 64)
            tl = wrap(seg["title"], tfont, W - 2 * PAD)[:3]
            layer = Image.new("RGBA", (W, len(tl) * 80 + 10), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            for i, line in enumerate(tl):
                ld.text((PAD, i * 80), line, font=tfont, fill=WHITE)
            self.title_layer = layer
            self.cap_y = self.TITLE_Y + layer.height + 36
        elif self.kind == "intro":
            self.cap_y = 1000 - self.cap_h // 2
        else:
            self.cap_y = 1000 - self.cap_h // 2

    # -- parts
    def _hero(self, frame: Image.Image, t: float) -> None:
        y, h = self.HERO_Y, self.HERO_H
        p = min(t / max(self.duration, 0.1), 1.0)
        if self.hero is not None:
            s = ZOOM - (ZOOM - 1.0) * p
            bw, bh = self.hero.size
            cw, ch = min(W * s, bw), min(h * s, bh)
            direction = 1 if self.index % 2 else -1
            left = max(0.0, (bw - cw) * (0.5 + direction * 0.25 * (p - 0.5)))
            top = max(0.0, (bh - ch) * 0.5)
            frame.paste(self.hero.resize((W, h), Image.BILINEAR, box=(left, top, left + cw, top + ch)), (0, y))
        else:
            d = ImageDraw.Draw(frame)
            bars, bw_, gap = 9, 70, 34
            x0 = (W - (bars * bw_ + (bars - 1) * gap)) // 2
            for i in range(bars):
                bh_ = 110 + 80 * (0.5 + 0.5 * math.sin(t * 3 + i * 0.9))
                if i == bars // 2:
                    bh_ = 380 * ease(t / 0.8) + 30 * math.sin(t * 5)
                x = x0 + i * (bw_ + gap)
                d.rectangle([x, y + h - 60 - bh_, x + bw_, y + h - 60], fill=SIGNAL_300 if i == bars // 2 else ACCENT_400)

    def _score(self, frame: Image.Image, t: float) -> None:
        """The memorable element: score counts up while an ECG trace draws and spikes."""
        score = self.seg.get("score")
        if not score:
            return
        d = ImageDraw.Draw(frame)
        color = tier_color(score)
        shown = score * ease((t - 0.1) / 0.8)
        y0 = 170
        d.text((PAD, y0), f"{shown:.1f}", font=font(True, 170), fill=color)
        nw = d.textlength(f"{score:.1f}", font=font(True, 170))
        d.text((PAD + nw + 14, y0 + 118), "/10", font=font(False, 40), fill="#8d98a5")
        # trace
        x_start, x_end, base = int(PAD + nw + 140), W - PAD - 40, y0 + 120
        width = x_end - x_start
        head = ease((t - 0.1) / 0.9)
        amp = 120 * (score / 10)
        pts = []
        for i in range(int(width * head) + 1):
            if i % 3 == 0:
                u = i / width
                pts.append((x_start + i, base - pulse_y(u, 1.0) * amp))
        if len(pts) > 1:
            d.line(pts, fill=color, width=6, joint="curve")
            hx, hy = pts[-1]
            d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], fill=WHITE)

    def _title(self, frame: Image.Image, t: float) -> None:
        a = ease((t - 0.2) / 0.35)
        layer = self.title_layer
        if a < 1.0:
            layer = layer.copy()
            layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
        frame.paste(layer, (0, self.TITLE_Y + int((1 - a) * 50)), layer)

    def _caption(self, frame: Image.Image, t: float, done: str, current: str, todo: str) -> None:
        d = ImageDraw.Draw(frame)
        progress = t / self.speech if self.speech > 0 else 1.0
        for i, (w, x, line) in enumerate(self.cap_words):
            start = self.cap_starts[i]
            nxt = self.cap_starts[i + 1] if i + 1 < len(self.cap_words) else 1.0
            color = done if progress >= nxt else current if progress >= start else todo
            d.text((PAD + x, self.cap_y + line * self.cap_line_h), w, font=self.cap_font, fill=color)

    def _bars(self, frame: Image.Image, t: float, base_y: int) -> None:
        d = ImageDraw.Draw(frame)
        bars, bw_, gap = 11, 56, 28
        x0 = (W - (bars * bw_ + (bars - 1) * gap)) // 2
        for i in range(bars):
            appear = ease((t - i * 0.04) / 0.4)
            h = (70 + 70 * (0.5 + 0.5 * math.sin(t * 3.2 + i * 0.8))) * appear
            color = ACCENT_400
            if i == bars // 2:
                h = (360 + 30 * math.sin(t * 4)) * ease((t - 0.2) / 0.5)
                color = SIGNAL_300
            d.rectangle([x0 + i * (bw_ + gap), base_y - h, x0 + i * (bw_ + gap) + bw_, base_y], fill=color)

    def frame(self, t: float) -> Image.Image:
        f = self.base.copy()
        if self.kind == "item":
            self._hero(f, t)
            self._score(f, t)
            self._title(f, t)
            self._caption(f, t, WHITE, SIGNAL_300, "#5d6874")
        elif self.kind == "intro":
            # Hook is on screen from frame 0: big words, image of the top story behind it
            if self.base is not None and self.seg.get("image") is None:
                self._bars(f, t, 700)
            self._caption(f, t, WHITE, SIGNAL_300, "#7d8996")
        else:
            self._bars(f, t, 700)
            self._caption(f, t, WHITE, SIGNAL_300, "#4a5f9a")

        total = max(self.global_total, 0.1)
        ImageDraw.Draw(f).rectangle([0, 0, int(W * min((self.global_start + t) / total, 1.0)), 10], fill=SIGNAL_300)

        # Punch-in at every cut: the frame lands 8% zoomed and settles in 0.2s
        k = 1.0 - ease(t / 0.2)
        if k > 0:
            z = 1.0 + 0.08 * k
            cw, ch = W / z, H / z
            l, tp = (W - cw) / 2, (H - ch) / 2
            f = f.resize((W, H), Image.BILINEAR, box=(l, tp, l + cw, tp + ch))
        return f
