#!/usr/bin/env python3
"""Render the README's ANSI-colored code blocks as PNG images.

GitHub's rendering of ```ansi fences is unreliable, so the colored sample
output is committed as pre-rendered PNGs instead.

Subcommands:

  extract  One-time migration: pull every fenced ```ansi block out of
           README.md, save the raw ANSI text under assets/ansi/<name>.txt,
           and replace each block in the README with an image reference to
           assets/<name>.png.

  render   Re-render assets/ansi/*.txt -> assets/*.png. Run this whenever
           an .txt source changes to refresh the images.

Usage:
    uv run python scripts/render_ansi_png.py extract
    uv run python scripts/render_ansi_png.py render
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ASSETS = ROOT / "assets"
ANSI_DIR = ASSETS / "ansi"

# Fenced ```ansi blocks appear in this order in the README.
BLOCKS: list[tuple[str, str]] = [
    ("token-legend", "Legend: green = green-list (key match), red = red-list (no match)"),
    ("token-watermarked", "Watermarked output, tokens colored by green-list match"),
    ("token-negative", "Negative-seed output evaluated against the detection seed"),
    ("token-plain", "Plain output without watermark"),
    ("synthid-legend", "Legend: hit strength 0-5 on a viridis color scale"),
    ("synthid-watermarked", "SynthID watermarked output colored by hit strength"),
    ("synthid-negative", "SynthID negative-seed output evaluated against the detection seed"),
    ("synthid-plain", "SynthID plain output without watermark"),
]

ANSI_FENCE = re.compile(r"```ansi\n(.*?)\n```", re.DOTALL)
SGR = re.compile(r"\x1b\[([0-9;]*)m")

# Rendering options.
FONT_SIZE = 15          # logical pixels
SCALE = 2               # supersample factor; final image is downscaled back
LINE_LEADING = 7        # extra logical pixels between lines
PADDING = 18            # logical pixels around the text block
BG = (13, 17, 23)       # GitHub dark canvas (#0d1117)
DEFAULT_FG = (230, 237, 243)  # GitHub dark foreground (#e6edf3)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/Library/Fonts/JetBrainsMono-Regular.ttf",
]


def load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Return (regular, bold) monospace fonts."""
    path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    if path is None:
        raise SystemExit("No monospace font found; add yours to FONT_CANDIDATES")
    size = FONT_SIZE * SCALE
    regular = ImageFont.truetype(path, size, index=0)
    family = regular.getname()[0]
    bold = None
    for i in range(8):
        try:
            candidate = ImageFont.truetype(path, size, index=i)
        except OSError:
            break
        if candidate.getname() == (family, "Bold"):
            bold = candidate
            break
    if bold is None:
        bold = regular  # fall back to stroke-based faux bold at draw time
    return regular, bold


def xterm256(n: int) -> tuple[int, int, int]:
    """Standard xterm-256 palette entry."""
    if n < 16:
        palette = [
            (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
            (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
            (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
        ]
        return palette[n]
    if n < 232:
        n -= 16
        levels = (0, 95, 135, 175, 215, 255)
        return (levels[n // 36], levels[(n // 6) % 6], levels[n % 6])
    g = 8 + 10 * (n - 232)
    return (g, g, g)


def parse_ansi(text: str) -> list[list[tuple[str, bool, int | None]]]:
    """Parse ANSI text into lines of (char, bold, 256-color) cells."""
    lines: list[list[tuple[str, bool, int | None]]] = []
    line: list[tuple[str, bool, int | None]] = []
    bold = False
    color: int | None = None
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "\x1b":
            m = SGR.match(text, pos)
            if m is None:  # not an SGR sequence we understand; drop it
                pos += 1
                continue
            params = m.group(1).split(";")
            i = 0
            while i < len(params):
                p = params[i]
                if p in ("", "0"):
                    bold, color = False, None
                elif p == "1":
                    bold = True
                elif p == "39":
                    color = None
                elif p == "38" and i + 2 < len(params) and params[i + 1] == "5":
                    color = int(params[i + 2])
                    i += 2
                # other SGR attributes are ignored
                i += 1
            pos = m.end()
            continue
        if ch == "\n":
            lines.append(line)
            line = []
        else:
            line.append((ch, bold, color))
        pos += 1
    lines.append(line)
    return lines


def render_png(ansi_text: str, out_path: Path) -> None:
    lines = parse_ansi(ansi_text.rstrip("\n"))
    regular, bold = load_fonts()

    # Monospace cell geometry (use the wider of the two faces).
    cell_w = max(regular.getlength("M"), bold.getlength("M"))
    ascent, descent = regular.getmetrics()
    line_h = ascent + descent + LINE_LEADING * SCALE

    def visible_len(cells: list[tuple[str, bool, int | None]]) -> int:
        n = len(cells)
        while n and cells[n - 1][0] == " ":
            n -= 1
        return n

    max_cols = max((visible_len(cells) for cells in lines), default=0)
    width = int(PADDING * 2 * SCALE + max_cols * cell_w) + SCALE
    height = int(PADDING * 2 * SCALE + len(lines) * line_h)

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    pad = PADDING * SCALE
    for row, cells in enumerate(lines):
        y = pad + row * line_h
        for col, (ch, is_bold, color) in enumerate(cells):
            if ch == " ":
                continue
            fill = xterm256(color) if color is not None else DEFAULT_FG
            font = bold if is_bold else regular
            stroke = 0
            if is_bold and bold is regular:
                stroke = max(1, SCALE)  # faux bold fallback
            draw.text(
                (pad + col * cell_w, y),
                ch,
                font=font,
                fill=fill,
                anchor="la",
                stroke_width=stroke,
                stroke_fill=fill,
            )

    img = img.resize((width // SCALE, height // SCALE), Image.LANCZOS)
    img.save(out_path)
    print(f"wrote {out_path.relative_to(ROOT)} ({img.width}x{img.height})")


def cmd_extract() -> None:
    text = README.read_text()
    blocks = ANSI_FENCE.findall(text)
    if len(blocks) != len(BLOCKS):
        sys.exit(f"expected {len(BLOCKS)} ```ansi blocks in README, found {len(blocks)}")
    ANSI_DIR.mkdir(parents=True, exist_ok=True)

    counter = iter(range(len(BLOCKS)))

    def replace(m: re.Match[str]) -> str:
        name, alt = BLOCKS[next(counter)]
        (ANSI_DIR / f"{name}.txt").write_text(m.group(1) + "\n")
        print(f"extracted {ANSI_DIR / name}.txt")
        return f"![{alt}](assets/{name}.png)"

    README.write_text(ANSI_FENCE.sub(replace, text))
    print(f"rewrote {README.name} with {len(blocks)} image references")


def cmd_render() -> None:
    for name, _ in BLOCKS:
        src = ANSI_DIR / f"{name}.txt"
        if not src.exists():
            sys.exit(f"missing {src}; run the extract subcommand first")
        render_png(src.read_text(), ASSETS / f"{name}.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract":
        cmd_extract()
    elif cmd == "render":
        cmd_render()
    else:
        sys.exit(__doc__)
