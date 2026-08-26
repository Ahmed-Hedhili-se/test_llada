# DMInfr — identity

**DM** — diffusion model · **Infr** — infrastructure.

The mark is an 8×10 token lattice shaped as a **D**. Density rises along the
diagonal: the upper-left field stays masked — three tokens are deliberately
absent — while the lower-right silhouette commits to full ink. Inside the
counter, four tokens step from a small violet seed to a full blue block: a
sequence being unmasked, not an arrow.

It is masked-diffusion decoding drawn literally, which is what this engine does.

```
   .  _  .  .  -              .   masked      outline only, undecided
   .  .  .  _  -  -           -   transitioning   violet, prediction in flight
   .  _        +  +           +   resolving   blue-leaning white
   .  .  1        +  #        #   resolved    committed token, full ink
   .  -     2     +  #        _   absent      still masked
   -  -        3  #  #      1-4   flow        violet -> blue, growing
   -  -           4  #  #
   -  -        #  #
   -  +  +  +  #  #
   -  +  +  +  #
```

## Files

| File | Use |
|---|---|
| `dminfr-mark.svg` | Primary. Full lattice, all states. Dark backgrounds, ≥48px |
| `dminfr-mark-inline.svg` | Compact, padded below so it optically centres beside `<h1>` type. `height="58"` |
| `dminfr-mark-compact.svg` | Solid silhouette, two flow tokens. 48px → 28px |
| `dminfr-mark-micro.svg` | 4×5 closed bowl, one flow token. Favicon, paper headers, <28px |
| `dminfr-mark-mono.svg` | Monochrome primary — papers, single-colour print |
| `dminfr-mark-micro-mono.svg` | Monochrome micro |
| `generate_logo.py` | Regenerates all of the above |

`-inline` exists because GitHub's markdown sanitizer strips `style` and
`valign`, so an `<img>` inside a heading can only sit on the text baseline —
which lines the mark's bottom edge up with the type's bottom edge and makes the
wordmark look like it is hanging off the corner. The `-inline` cut carries empty
space below the lattice instead, so the mark centres against the cap height.

It is cut from `compact`, not `primary`, for the reason the size ladder gives: a
README header renders the mark at roughly 46px, and at that size the full
lattice's masked field — 26%-opacity outlines — reads as grey static instead of
a letter. The solid silhouette reads as a **D** instantly.

The `-mono` files paint with `currentColor`, so inlined in HTML they inherit
the surrounding text colour. Referenced through `<img src>` they fall back to
black, which is the intended behaviour for print.

## Rules

- **Clearspace** — two token units on every side.
- **Never recolour the wordmark.** Type is monochrome; colour lives only in the
  flow path.
- **Never rotate the lattice.** The diagonal carries the meaning.
- Below 28px use the micro cut. The 8×10 lattice turns to mush.

## Typography

IBM Plex Sans. **DM** at 600, *Infr* at 300, tracking −3.5%. IBM Plex Mono for
labels, code and terminal output at +20% tracking.

## Palette

| | Hex | Role |
|---|---|---|
| near-black | `#0B0C0E` | background |
| graphite | `#22252B` | masked |
| violet | `#8B5CF6` | transitioning |
| deep blue | `#1D4ED8` | flow end |
| ink white | `#F5F6F8` | resolved |

## Regenerating

```bash
python assets/logo/generate_logo.py
```

Source of truth is the Claude Design project *DMInfr Logo Design*
(`DMInfrMark.dc.html`), which renders through a React runtime. `generate_logo.py`
reproduces the same lattice as static SVG so the repo carries no runtime
dependency — the lattice, flow path, drop set and density ramp are copied
verbatim from the canvas component. If the canvas mark changes, update the
constants at the top of the script.

One deliberate difference: the canvas mixes colours in oklab, the script
interpolates in sRGB. Across four discrete flow steps the difference is not
visible, and it keeps the generator dependency-free.
