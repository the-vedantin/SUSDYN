# DESIGN.md — Vahan UI design system (Impeccable)

The single source of truth for Vahan's fonts and font colors. Recovered from the
binder styling we did "the impeccable way" (`DESIGN_2027/BINDER_2027_v2/make_html_binder.py`,
CSS `:root` block) and adapted for the app's DARK theme. Format follows Impeccable
(impeccable.style) — a DESIGN.md that travels with the project so the system never
rots again.

Scope of this file: **fonts and font/text colors only.** Graph LINE colors
(the FL/FR/RL/RR corner curves) are deliberately OUT of scope and unchanged.

## Font

One family, the Impeccable system stack (no proprietary font to install):

```
'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif
```

Monospace (numeric readouts, tables): `'Consolas', 'SF Mono', monospace`.

Base size 13px, line-height 1.5. Numbers that line up in columns use
`font-variant-numeric: tabular-nums` (Qt: a monospace font).

## Color (dark-theme adaptation of the Impeccable tokens)

| Token | Light binder | Dark app | Use |
|---|---|---|---|
| `ink`   | `#1b1b1e` | `#ECECEE` | primary text |
| `muted` | `#6d6d72` | `#9A9AA2` | secondary text, captions, table headers |
| `paper` | `#faf9f6` | `#0B0B0D` | window background |
| `panel` | `#ffffff` | `#141416` | cards / raised surfaces |
| `line`  | `#e5e2dc` | `#2A2A2E` | hairline rules, borders |
| `accent`| `#cc1f2d` (red) | `#E23B48` (red) | eyebrows, emphasis, primary buttons |

**YELLOW IS BANNED FROM TEXT/UI.** The user hates it and is colorblind
(yellow/orange confusion). Every `#FFD600` / amber (`#FFB74D`, `#FFB300`,
`#b8860b`, `#D99000`) used as a TEXT or UI color is replaced by `accent` red or
`ink`/`muted`. (Yellow may still appear as a GRAPH LINE color — that is a
separate palette the user did not ask to change.)

## Type scale & rules

- **Eyebrow / category label**: uppercase, `letter-spacing: 0.15em`, weight 650,
  color `accent`, ~11px. (Impeccable `.cat`.)
- **Section header**: weight 600, `ink`, no yellow.
- **Body**: 13px, `ink`, line-height 1.5.
- **Caption / sub**: 11–12px, `muted`.
- **Table header**: uppercase, 11px, `letter-spacing: 0.06em`, `muted`, one firm
  `ink`/`line` rule under the header row, quiet hairline rows (the "impeccable
  table"). Right-align numbers, tabular figures.
- Headlines get tight tracking (`letter-spacing: -0.02em`) and balanced wrap.

## Principles (from Impeccable)

- Respect the existing system: inherit tokens, don't invent one-off colors.
- No purple gradients, no glassmorphism, no vague filler.
- Every number gets a plain-language label of **what it does**, not a verdict.
