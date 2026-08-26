"""Generate the DMInfr mark as dependency-free SVG.

The mark is authored in Claude Design as `DMInfrMark.dc.html`, which renders
through a React runtime (`support.js`, ~70 KB). That is fine for a design
canvas and wrong for an inference repo, so this script reproduces the same
lattice as static SVG: no runtime, no build step, embeddable in a README, a
paper, or a favicon.

The spec, from the design:

    An 8x10 token lattice shaped as a "D". Density rises along the diagonal --
    the upper-left field stays masked (three tokens deliberately absent) while
    the lower-right silhouette commits to full ink. Inside the counter, four
    tokens step from a small violet seed to a full blue block: a sequence being
    unmasked, not an arrow. Colour lives only in that path.

Which is masked-diffusion decoding drawn literally: masked tokens, tokens in
flight, tokens resolved.

Run:  python assets/logo/generate_logo.py
"""

from __future__ import annotations

import os
import re

# --- the lattice, verbatim from DMInfrMark.dc.html -------------------------

MAP = [
    "#####...",
    "######..",
    "##...##.",
    "##....##",
    "##....##",
    "##....##",
    "##....##",
    "##...##.",
    "######..",
    "#####...",
]
MICRO = ["###.", "#.##", "#..#", "#.##", "###."]

#: (row, col) of the four flow tokens inside the counter.
FLOW = [(3, 2), (4, 3), (5, 4), (6, 5)]
#: Each flow token is drawn smaller than its cell, growing along the path.
SCALES = [0.32, 0.52, 0.74, 1.0]
#: Deliberately absent tokens -- the "masked" field in the upper left.
DROPS = {(0, 1), (1, 3), (2, 1)}

# --- palette ---------------------------------------------------------------

INK = "#F5F6F8"        # resolved   -- committed token, full ink
RESOLVING = "#CED7F0"  # blue-leaning white, drawn at 74% alpha
ACCENT_A = "#8B5CF6"   # violet -- transitioning, prediction in flight
ACCENT_B = "#1D4ED8"   # deep blue -- flow end
BG = "#0B0C0E"

# --- geometry --------------------------------------------------------------
# Cell and gap sizes are derived from the design's percentage gaps so the
# proportions match the canvas exactly: 1.94% column gap / 1.55% row gap on the
# full lattice gives a 0.797 aspect ratio with square cells.

FULL = dict(cols=8, rows=10, cell=10.8, gap_x=1.94, gap_y=1.94, radius_pct=0.08)
MICRO_GEO = dict(cols=4, rows=5, cell=23.26, gap_x=2.33, gap_y=2.32, radius_pct=0.06)


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Blend two hex colours. The canvas mixes in oklab; sRGB is close enough
    for four discrete steps and keeps this file dependency-free."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _canvas(geo):
    w = geo["cols"] * geo["cell"] + (geo["cols"] - 1) * geo["gap_x"]
    h = geo["rows"] * geo["cell"] + (geo["rows"] - 1) * geo["gap_y"]
    return round(w, 2), round(h, 2)


def _rect(x, y, size, radius, **attrs):
    parts = [f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}"',
             f'rx="{radius:.2f}"']
    for k, v in attrs.items():
        parts.append(f'{k.replace("_", "-")}="{v}"')
    return "  " + " ".join(parts) + "/>"


def _token_level(r: int, c: int) -> int:
    """Density along the diagonal: 0 masked, 1 transitioning, 2 resolving,
    3 resolved. Weighted 55% by column, 45% by row -- the same ramp the canvas
    uses, which is what makes the mark read as a gradient of certainty rather
    than a checkerboard."""
    p = 0.55 * (c / 7) + 0.45 * (r / 9)
    return 0 if p < 0.24 else 1 if p < 0.46 else 2 if p < 0.7 else 3


def build_full(mono: bool = False, solid: bool = False) -> str:
    """Primary (8x10) mark.

    mono  -- every token uses currentColor at varying alpha, for papers and
             single-colour printing.
    solid -- the `compact` variant: full silhouette, no masked field, only the
             two largest flow tokens. Legible down to ~28px.
    """
    geo = FULL
    w, h = _canvas(geo)
    cell, gx, gy = geo["cell"], geo["gap_x"], geo["gap_y"]
    radius = cell * geo["radius_pct"]
    out = []

    for r in range(geo["rows"]):
        for c in range(geo["cols"]):
            x, y = c * (cell + gx), r * (cell + gy)
            fi = FLOW.index((r, c)) if (r, c) in FLOW else -1

            # Flow tokens first: they sit inside the counter, where MAP is
            # empty, so they are drawn regardless of the silhouette.
            if fi >= 0 and not (solid and fi < 2):
                k = fi / (len(FLOW) - 1)
                colour = "currentColor" if mono else _lerp_hex(ACCENT_A, ACCENT_B, k)
                s = 1.0 if solid else SCALES[fi]
                size = cell * s
                off = (cell - size) / 2
                opacity = "1" if solid else f"{0.62 + 0.38 * k:.2f}"
                out.append(_rect(x + off, y + off, size, radius * s,
                                 fill=colour, opacity=opacity))
                continue

            if MAP[r][c] != "#":
                continue
            if not solid and (r, c) in DROPS:
                continue  # deliberately absent -- still masked

            level = 3 if solid else _token_level(r, c)
            if level == 0:
                # Masked: outline only. Present but undecided.
                out.append(_rect(x + 0.5, y + 0.5, cell - 1, radius, fill="none",
                                 stroke="currentColor" if mono else INK,
                                 stroke_width="1", opacity="0.26"))
            elif mono:
                out.append(_rect(x, y, cell, radius, fill="currentColor",
                                 opacity=["0", "0.26", "0.58", "1"][level]))
            elif level == 1:
                out.append(_rect(x, y, cell, radius, fill=ACCENT_A, opacity="0.42"))
            elif level == 2:
                out.append(_rect(x, y, cell, radius, fill=RESOLVING, opacity="0.74"))
            else:
                out.append(_rect(x, y, cell, radius, fill=INK))

    label = "monochrome" if mono else ("compact" if solid else "primary")
    return _svg(w, h, out, f"DMInfr mark ({label})")


def build_micro(mono: bool = False) -> str:
    """4x5 lattice: closed bowl, tight gaps, one resolved flow token.

    Below ~28px the 8x10 lattice turns to mush, so the micro cut drops to a
    silhouette with a single colour accent. Favicons and paper headers.
    """
    geo = MICRO_GEO
    w, h = _canvas(geo)
    cell, gx, gy = geo["cell"], geo["gap_x"], geo["gap_y"]
    radius = cell * geo["radius_pct"]
    out = []

    for r in range(geo["rows"]):
        for c in range(geo["cols"]):
            x, y = c * (cell + gx), r * (cell + gy)
            if (r, c) == (2, 2):
                # The single flow token, carrying the whole diffusion idea at
                # 16px. Sits in the counter, which MICRO leaves empty.
                colour = "currentColor" if mono else _lerp_hex(ACCENT_A, ACCENT_B, 0.55)
                out.append(_rect(x, y, cell, radius, fill=colour))
            elif MICRO[r][c] == "#":
                out.append(_rect(x, y, cell, radius,
                                 fill="currentColor" if mono else INK))

    return _svg(w, h, out, "DMInfr mark (micro)")


def _svg(w: float, h: float, body: list, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"\n'
        f'     width="{w}" height="{h}" role="img" aria-label="{title}"\n'
        f'     fill="none">\n'
        f'  <title>{title}</title>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


# --- inline lockup ---------------------------------------------------------
# GitHub's markdown sanitizer strips `style` and `valign`, so an <img> placed
# inside an <h1> can only sit on the text baseline: the mark's bottom edge and
# the wordmark's bottom edge line up, and the type appears to hang off the
# lower corner of the logo. No attribute GitHub allows will fix that.
#
# So bake the offset into the file. The `-inline` cut carries empty space below
# the lattice, sized so that when rendered at the height this script prints,
# the mark's optical centre lands on the wordmark's cap centre.
#
#   GitHub's h1 is 2em = 32px, cap height ~= 0.72em ~= 23px above the baseline.
#   For a mark V px tall, (V - cap)/2 of it has to fall *below* the baseline --
#   which is exactly how much empty space the file needs at its bottom edge.

#: Cap height of GitHub's <h1>, in px. See above.
H1_CAP_PX = 23.0


def build_inline(source: str, visible_px: float, cap_px: float = H1_CAP_PX):
    """Baseline-compensate `source`. Returns (svg, render_height_px)."""
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', source)
    w, h = float(vb.group(1)), float(vb.group(2))

    below_px = (visible_px - cap_px) / 2   # mark height that must sit below the baseline
    pad = below_px * (h / visible_px)      # the same distance, in viewBox units
    total = round(h + pad, 2)

    out = source.replace('viewBox="0 0 %s %s"' % (vb.group(1), vb.group(2)),
                         'viewBox="0 0 %s %s"' % (vb.group(1), total))
    out = re.sub(r'width="[\d.]+" height="[\d.]+"',
                 'width="%s" height="%s"' % (w, total), out, count=1)
    out = out.replace("(compact)", "(inline)")
    return out, round(visible_px + below_px, 1)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    primary = build_full()
    # The inline cut is built from `compact`, not `primary`. At the ~48px a
    # README header gives it, the full lattice's masked field (26% outlines)
    # reads as grey static rather than a letter -- which is the whole reason
    # the compact silhouette exists. See the size ladder in README.md.
    inline, inline_h = build_inline(build_full(solid=True), visible_px=46.0)
    files = {
        "dminfr-mark.svg": primary,
        "dminfr-mark-inline.svg": inline,
        "dminfr-mark-mono.svg": build_full(mono=True),
        "dminfr-mark-compact.svg": build_full(solid=True),
        "dminfr-mark-micro.svg": build_micro(),
        "dminfr-mark-micro-mono.svg": build_micro(mono=True),
    }
    for name, svg in files.items():
        path = os.path.join(here, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
        print(f"  wrote {name:30s} {len(svg):5d} bytes")
    print()
    print(f'  dminfr-mark-inline.svg -> height="{inline_h:.0f}" inside a GitHub <h1>')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
