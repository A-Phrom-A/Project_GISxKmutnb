# GISlab True Miracle — Presentation / Infographic Design System

Scope: infographic-style pitch deck slides (image-generated or built in Figma/Canva/Keynote)
Companion to: `design-web.md` (shares color, typography, spacing, radius, and animation tokens)
Status: v0.2 (Aug 2026)

---

## 1. Overview & Aesthetic Direction

Slides should feel like **approachable, playful, yet modern data dashboard screenshots**, directly matching the aesthetic of the web platform:
- White/light canvas for General, Agri, and Utility sections.
- Neutral dark slate canvas for Disaster Risk / active alert sections to emphasize warning states.
- Highly rounded bento blocks, emoji icons, and restrained color accents used to group information by sector.
- Inspired by **shadcn/ui** conventions—clean spacing, soft drop shadows, and minimalist borders.

---

## 2. Canvas & Grid

- **Canvas**: 1920 × 1080 (16:9)
- **Safe margin**: 96px all sides
- **Content grid**: 12-column, 24px gutter — same bento logic as the web system, scaled for presentation readability.
- **Card sizing**: Reuse web grid variants (`col-4` for KPIs, `col-4` for feeds, `col-8` for hero map visuals).

### 2.1 Allowed Column Classes (Strict Whitelist)

> [!CAUTION]
> **Only the following column classes are defined in CSS.** Using any class outside this list (e.g., `col-5`, `col-7`, `col-9`, `col-10`, `col-11`) will cause the browser to fall back to a 1-column width, breaking the layout.

| Class | Span | Common Use |
|---|---|---|
| `.col-3` | 3/12 (25%) | Small KPI card, compact stat |
| `.col-4` | 4/12 (33%) | Standard KPI card, alert feed, sidebar panel |
| `.col-6` | 6/12 (50%) | Half-width card, comparison pair |
| `.col-8` | 8/12 (67%) | Hero map canvas, main chart area |
| `.col-12` | 12/12 (100%) | Full-width banner, cover slide content |

**Valid row combinations (must sum to exactly 12):**
- `col-4 + col-4 + col-4` (3 equal cards)
- `col-8 + col-4` (hero + sidebar)
- `col-4 + col-8` (sidebar + hero)
- `col-6 + col-6` (equal split)
- `col-3 + col-3 + col-3 + col-3` (4 compact cards)
- `col-12` (full-width single card)

---

## 3. Color Usage on Slides

Aligns directly with the colors defined in `design-web.md`:

### Light Slides (General / Agri / Utility)
- **Base Canvas**: `--bg` (#F3F8F5)
- **Card Surface**: `--bg-card` (#FFFFFF)
- **Text**: `--text-h` (#1B2B22) and `--text-body` (#3D5247)

### Dark Slides (Disaster / Active Alerts)
- **Base Canvas**: `--bg` (#131418)
- **Card Surface**: `--bg-card` (#1C1E24)
- **Text**: `--text-h` (#F0F2F5) and `--text-body` (#A8AEBA)

### Accent Allocations

Accent colors should be applied flexibly based on the semantic meaning and context of the slide content to guide the viewer's focus:
- **Semantic Accents**: Apply green, blue, amber, or red accents dynamically depending on what the data represents (e.g., green for healthy growth or success metrics, blue for utility assets or water bodies, amber for warnings or degradation trends, and red for active anomalies, critical disasters, or emergency flags).
- **Flexible Mapping**: Allow accents to adapt to the slide's primary topic rather than locking sectors strictly to a single color, ensuring the color code directly serves the slide's narrative.

---

## 4. Typography for Slides

Font stack: `"Mitr"` ➔ `"Kanit"` ➔ default sans-serif fallback (same as web)

| Style | Size (px) | Weight | Notes |
|---|---|---|---|
| Slide title | 56px | 600 | Mitr |
| Slide subtitle | 24px | 400 | Mitr / Kanit |
| Card title | 22px | 600 | Mitr |
| Card body | 16px | 400 | Mitr |
| Stat / big number | 68px | 700 | Mitr (bold numerical indicator) |
| Caption / source | 13px | 500 | `text-muted` |

---

## 5. Spacing, Radius & Elevation

To replicate the playful bento widget feel, use the exact tokens from the web platform:

- **Border Radius**: `--r-md` (20px) for small badges, `--r-lg` (28px) for cards, `--r-xl` (36px) for main layouts.
- **Elevation**: `--shadow-md` for resting cards, transitioning to `--shadow-lg` for focal cards.

### 5.1 Slide Overflow Prevention (Spacing Budget)

The total available content height inside a 1080px slide is limited. Exceeding the budget causes cards to overflow beyond the slide boundary.

| Element | Max Value | Notes |
|---|---|---|
| **Slide padding** (top + bottom) | `56px` top + `40px` bottom = **96px** | Do not exceed. Use `padding: 56px 80px 40px;` |
| **Slide header** (title + subtitle) | **~80px** total | Title: `2.2rem`, subtitle: `1rem`. Keep `margin-bottom: 16px` max |
| **Bento grid gap** | `20px` | Standard gap between cards |
| **Card internal padding** | `24px` max | Do not stack heavy margins inside cards |
| **Remaining for card content** | **~864px** | `1080 - 96 (slide pad) - 80 (header) - 40 (footer)` |

> [!WARNING]
> **Common overflow causes:**
> - Setting slide padding to `80px` on all 4 sides (wastes 160px of vertical space)
> - Large `margin-bottom` on slide title (e.g., `28px` or more)
> - Heavy card internal padding (`32px+`) combined with tall content
> - Using `font-size: 3rem+` for card content text
>
> **Rule of thumb:** If cards contain more than 2 rows of bento grid, reduce card padding to `16px` and slide padding to `48px 60px 32px`.

---

## 6. Icon & Illustration Style

- **Icons**: Prefer SVG line icons or Unicode symbols. Do not use emoji unless strictly necessary for prototyping. If emoji is present, add a comment `<!-- TODO: Replace emoji with SVG icon -->` to flag for future cleanup.
- **Line Icons**: Soft, rounded stroke icons (1.5–2.0px stroke width).
- **Geospatial motifs**: Minimalist vector patterns (satellite vectors, parcel grids, water contours, road networks).
- No generic photographic textures. Keep illustrations vector-based or direct web app mockups.

---

## 7. Slide Templates
 
 1. **Cover** — Clean white canvas, logo emoji `🌍`, large headline, and subtle spatial texture.
 2. **Section Divider** — Clean solid canvas in theme color, with large sector label (Agri / Utility / Disaster) in the sector's accent color.
 3. **Asymmetric KPI Bento** — 3 varied KPI cards displaying large bold metrics and contextual badges.
 4. **Spatial Canvas Showcase** — Edge-to-edge Deck.GL v9 map frame (`col-8`) with floating frosted-glass controls + `col-4` live AI Signals feed.
 5. **Analytical Split** — `col-4` Change Detection stat rings paired with `col-8` NDVI health trend charts.

---

## 8. Motion & Transition (for Live / Interactive Demos)

- Follow the web transition model:
  - Springy entry animations: `--t-med: 0.38s cubic-bezier(0.34,1.56,0.64,1)` for slides and card hover effects.
  - Quick fades: `--t-fast: 0.16s ease`.

---

## 9. Do's and Don'ts

**Do:**
- Theme selection (Light vs Dark) is a user preference. Recommend light for Agri/Utility and dark for Disaster as defaults, but allow override based on user preference and presentation context.
- Keep border radii consistent with the web widgets.
- Use icons strategically to guide visual interest. Prefer SVG line icons over emoji.
- Stagger animations of cards so they slide in sequence.
- **Always verify** that column classes used in HTML exist in the CSS whitelist (Section 2.1).
- **Check total vertical spacing** before finalizing a slide to prevent overflow (Section 5.1).

**Don't:**
- Use highly saturated primary colors for card backgrounds.
- Use green-tinted dark backgrounds (maintain neutral dark slate `#131418`).
- Show raw unsmoothed maps (use Deck.GL vector snapshot style).
- Use emoji in final production slides without flagging for replacement.
- **Never use undefined column classes** (e.g., `col-5`, `col-7`, `col-9`, `col-10`, `col-11`).
- **Never set slide padding above 80px** per side — it directly causes card overflow on dense slides.
