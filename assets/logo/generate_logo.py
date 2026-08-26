"""Generate the DMInfr mark and lockup as dependency-free SVG.

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

Every cut ships in a dark-ground and a light-ground version. "Resolved" means
maximum contrast against the page, so it is ink white on dark and near-black on
light -- not one file with its opacity turned down.

Run:  python assets/logo/generate_logo.py
"""

from __future__ import annotations

import os

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

# --- palettes --------------------------------------------------------------
# The accent path is identical on both grounds: colour is the one thing that
# should not change between themes, or the mark stops being the same mark.

ACCENT_A = "#8B5CF6"   # violet -- transitioning, prediction in flight
ACCENT_B = "#1D4ED8"   # deep blue -- flow end

#: On a dark ground, "resolved" is ink white and the ramp brightens toward it.
DARK = dict(
    ink="#F5F6F8",         # resolved -- committed token
    resolving="#CED7F0",   # blue-leaning white
    resolving_op="0.74",
    trans_op="0.42",       # violet, still translucent
    masked_op="0.26",      # outline only
    text="#F5F6F8",
)

#: On a light ground the ramp darkens instead. Violet at 45% reads as the same
#: lavender the dark cut reads as at 42% -- the accent survives the inversion.
LIGHT = dict(
    ink="#16181D",
    resolving="#2F3648",
    resolving_op="0.80",
    trans_op="0.45",
    masked_op="0.20",
    text="#16181D",
)

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
    return " ".join(parts) + "/>"


def _token_level(r: int, c: int) -> int:
    """Density along the diagonal: 0 masked, 1 transitioning, 2 resolving,
    3 resolved. Weighted 55% by column, 45% by row -- the same ramp the canvas
    uses, which is what makes the mark read as a gradient of certainty rather
    than a checkerboard."""
    p = 0.55 * (c / 7) + 0.45 * (r / 9)
    return 0 if p < 0.24 else 1 if p < 0.46 else 2 if p < 0.7 else 3


def lattice_full(pal=DARK, mono: bool = False, solid: bool = False,
                 dx: float = 0.0, dy: float = 0.0) -> list:
    """The 8x10 lattice as a list of <rect> strings, offset by (dx, dy).

    pal   -- DARK or LIGHT. Decides what "resolved" means.
    mono  -- every token uses currentColor at varying alpha, for papers and
             single-colour printing.
    solid -- the `compact` variant: full silhouette, no masked field, only the
             two largest flow tokens. Legible down to ~28px.
    """
    geo = FULL
    cell, gx, gy = geo["cell"], geo["gap_x"], geo["gap_y"]
    radius = cell * geo["radius_pct"]
    out = []

    for r in range(geo["rows"]):
        for c in range(geo["cols"]):
            x, y = dx + c * (cell + gx), dy + r * (cell + gy)
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
                                 stroke="currentColor" if mono else pal["ink"],
                                 stroke_width="1", opacity=pal["masked_op"]))
            elif mono:
                out.append(_rect(x, y, cell, radius, fill="currentColor",
                                 opacity=["0", pal["masked_op"],
                                          pal["resolving_op"], "1"][level]))
            elif level == 1:
                out.append(_rect(x, y, cell, radius, fill=ACCENT_A,
                                 opacity=pal["trans_op"]))
            elif level == 2:
                out.append(_rect(x, y, cell, radius, fill=pal["resolving"],
                                 opacity=pal["resolving_op"]))
            else:
                out.append(_rect(x, y, cell, radius, fill=pal["ink"]))

    return out


def build_full(pal=DARK, mono: bool = False, solid: bool = False) -> str:
    w, h = _canvas(FULL)
    label = "monochrome" if mono else ("compact" if solid else "primary")
    return _svg(w, h, lattice_full(pal, mono, solid), f"DMInfr mark ({label})")


def build_micro(pal=DARK, mono: bool = False) -> str:
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
                                 fill="currentColor" if mono else pal["ink"]))

    return _svg(w, h, out, "DMInfr mark (micro)")


# --- lockup ----------------------------------------------------------------
# Mark and wordmark on one optical centre line.
#
# The wordmark is SVG <text>, not outlines: converting IBM Plex to paths needs
# the font binary and fontTools, and pinning a 40 KB font subset into every
# consumer of this repo is a worse trade than a fallback stack. To stop the
# lockup's geometry from moving when a viewer lacks Plex, each word is given an
# explicit `textLength` with `lengthAdjust="spacing"` -- glyph shapes are left
# alone and only the tracking absorbs the difference, so the lockup occupies
# the same box in every renderer.

MARK_W, MARK_H = _canvas(FULL)
#: Gap between mark and wordmark. The identity's clearspace unit is one cell.
LOCKUP_GAP = FULL["cell"] * 2.6
FONT_PX = 54.0
#: Locked advance widths, so a fallback face cannot reflow the lockup.
DM_LEN, INFR_LEN = 88.0, 92.0
FONT_STACK = ("'IBM Plex Sans','Segoe UI',-apple-system,BlinkMacSystemFont,"
              "Helvetica,Arial,sans-serif")


def build_lockup(pal=DARK) -> str:
    text_x = MARK_W + LOCKUP_GAP
    # Centre the cap height on the mark's centre. Cap height is ~0.72em for
    # every face in the stack, so the baseline sits half a cap below centre.
    baseline = MARK_H / 2 + (FONT_PX * 0.72) / 2
    width = round(text_x + DM_LEN + INFR_LEN, 2)

    body = lattice_full(pal)
    body.append(
        f'<text x="{text_x:.2f}" y="{baseline:.2f}" fill="{pal["text"]}"'
        f' font-family="{FONT_STACK}" font-size="{FONT_PX:.0f}"'
        f' letter-spacing="-1.9">'
        f'<tspan font-weight="600" textLength="{DM_LEN:.0f}" '
        f'lengthAdjust="spacing">DM</tspan>'
        f'<tspan font-weight="300" textLength="{INFR_LEN:.0f}" '
        f'lengthAdjust="spacing">Infr</tspan>'
        f'</text>'
    )
    return _svg(width, MARK_H, body, "DMInfr")


def _svg(w: float, h: float, body: list, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"\n'
        f'     width="{w}" height="{h}" role="img" aria-label="{title}"\n'
        f'     fill="none">\n'
        f'  <title>{title}</title>\n'
        + "\n".join("  " + line for line in body) + "\n</svg>\n"
    )


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    files = {
        # lockups -- what a README or a slide should use
        "dminfr-lockup.svg": build_lockup(DARK),
        "dminfr-lockup-light.svg": build_lockup(LIGHT),
        # marks
        "dminfr-mark.svg": build_full(DARK),
        "dminfr-mark-light.svg": build_full(LIGHT),
        "dminfr-mark-compact.svg": build_full(DARK, solid=True),
        "dminfr-mark-compact-light.svg": build_full(LIGHT, solid=True),
        "dminfr-mark-micro.svg": build_micro(DARK),
        "dminfr-mark-micro-light.svg": build_micro(LIGHT),
        # monochrome -- currentColor, so the ground is the caller's problem
        "dminfr-mark-mono.svg": build_full(DARK, mono=True),
        "dminfr-mark-micro-mono.svg": build_micro(DARK, mono=True),
    }
    for name, svg in files.items():
        path = os.path.join(here, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
        print(f"  wrote {name:34s} {len(svg):5d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
