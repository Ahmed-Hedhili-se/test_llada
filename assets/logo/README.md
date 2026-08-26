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

Every cut ships in a dark-ground and a light-ground version. **Resolved means
maximum contrast against the page** — ink white on dark, near-black on light —
so the two are separate files, not one file with its opacity turned down. The
accent path is byte-identical in both: colour is the one thing that must not
change between themes, or it stops being the same mark.

| File | Use |
|---|---|
| `dminfr-lockup.svg` · `-light` | **Mark + wordmark.** What a README, slide, or paper header should use |
| `dminfr-mark.svg` · `-light` | Primary. Full lattice, all four states. ≥64px |
| `dminfr-mark-compact.svg` · `-light` | Solid silhouette, two flow tokens. 64px → 28px |
| `dminfr-mark-micro.svg` · `-light` | 4×5 closed bowl, one flow token. Favicon, <28px |
| `dminfr-mark-mono.svg` | Monochrome primary — single-colour print |
| `dminfr-mark-micro-mono.svg` | Monochrome micro |
| `generate_logo.py` | Regenerates all of the above |

Pick per theme with `<picture>`, which GitHub honours in Markdown:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/dminfr-lockup.svg">
    <img src="assets/logo/dminfr-lockup-light.svg" alt="DMInfr" width="340">
  </picture>
</p>
```

The `-mono` files paint with `currentColor`, so inlined in HTML they inherit
the surrounding text colour. Referenced through `<img src>` they fall back to
black, which is the intended behaviour for print.

### The wordmark is live text, not outlines

Converting IBM Plex to paths needs the font binary and `fontTools`, and pinning
a font subset into every consumer of this repo is a worse trade than a fallback
stack. So the lockup's wordmark is SVG `<text>`. To stop a missing Plex from
reflowing the lockup, each word carries an explicit `textLength` with
`lengthAdjust="spacing"`: glyph shapes are untouched and only the tracking
absorbs the difference, so the lockup occupies the same box in every renderer.

If you need a guaranteed-identical wordmark — a printed paper, a conference
slide template — export the lockup to outlines once and commit that file
alongside these.

## Rules

- **Clearspace** — two token units on every side.
- **Never recolour the wordmark.** Type is monochrome; colour lives only in the
  flow path.
- **Never rotate the lattice.** The diagonal carries the meaning.
- **Never put a dark-ground cut on a light page.** Resolved tokens are ink white;
  they vanish. Use the `-light` file or `<picture>`.
- **Size ladder** — primary at 64px and up, compact from 64px down to 28px, micro
  below 28px. The floor on the primary was 48px until it was tried at 48px: the
  masked field is drawn as 26%-opacity outlines, and those degrade into grey
  static well before the silhouette does.

## Typography

IBM Plex Sans. **DM** at 600, *Infr* at 300, tracking −3.5%. IBM Plex Mono for
labels, code and terminal output at +20% tracking.

## Palette

The accent path is shared. Only the ends of the ramp swap.

| | Hex | Role |
|---|---|---|
| violet | `#8B5CF6` | transitioning — **both grounds** |
| deep blue | `#1D4ED8` | flow end — **both grounds** |
| ink white | `#F5F6F8` | resolved, on dark |
| pale slate | `#CED7F0` | resolving, on dark |
| near-black | `#16181D` | resolved, on light |
| slate | `#2F3648` | resolving, on light |
| near-black | `#0B0C0E` | dark ground |
| graphite | `#22252B` | rules, panels |

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
