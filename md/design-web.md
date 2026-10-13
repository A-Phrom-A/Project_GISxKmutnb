# True Miracle — Web Application Design System

Product: AI-Powered Intelligence Platform (Agriculture / Utility / Disaster Monitoring)  
Scope: Web Application / Interactive Dashboard — Vanilla CSS + Deck.GL v9 + MapLibre GL  
Status: v0.3 (Production Standard)

---

## 1. Overview & Architectural Direction

This design system establishes the visual and functional standards for the **True Miracle** web application. The visual language blends a clean, modern SaaS aesthetic inspired by **shadcn/ui** with high-performance WebGL data visualization powered by **Deck.GL v9**.

### Core Architecture Highlights
- **Multi-Domain & Scenario-Driven Model**: Dynamic state management across domains (Agriculture, Utility, Disaster) and specialized operational scenarios.
- **Dynamic CSS Variable Theming**: Centralized CSS custom properties with live accent switching (`--accent-domain`, `--accent-domain-d`, `--accent-domain-xl`).
- **Edge-to-Edge Interactive Canvas**: Map and data visualizations seamlessly fill rounded Bento containers with floating controls, avoiding rigid legacy desktop toolbars.
- **Client-Side Resilience**: Self-contained inline raster basemap definitions (`MapLibre GL`) to prevent CORS constraints in local and offline environments.

## 2. Design Principles

1. **shadcn/ui Conventions**: Adopt unstyled, clean, and accessible component behaviors, styled with a modern minimalist aesthetic.
2. **Modern & Playful Widgets**: Highly rounded corners (`--r-*`) and soft, diffused elevations (`--shadow-*`) to give a friendly yet professional tone.
3. **Bento Grid Layout**: Data is organized inside a structured 12-column grid system.
4. **Airy & Borderless Hierarchy**: Avoid "boxes inside boxes" (no bulky pill frames or heavy nested card bubbles). Use generous whitespace, clean typography hierarchy, and subtle hairline dividers (`1px solid var(--border)`) instead of encasing list items in inner cards.
5. **Geospatial Focus**: Built on top of **Deck.GL v9** for interactive, vector-like layer visualization.

---

## 3. Color System & Design Tokens

### 3.1 Base Surface Tokens

| CSS Variable | Light Theme | Dark Theme | Usage |
|---|---|---|---|
| `--bg` | `#F3F8F5` | `#131418` | Main viewport background |
| `--bg-card` | `#FFFFFF` | `#1C1E24` | Bento card surface |
| `--bg-sidebar` | `#FAFCFB` | `#111217` | Navigation sidebar |
| `--bg-hover` | `#EDF5F0` | `#24262E` | Hover states and background fills |
| `--text-h` | `#1B2B22` | `#F0F2F5` | Headings and prominent titles |
| `--text-body` | `#3D5247` | `#A8AEBA` | Primary body and narrative copy |
| `--text-muted` | `#7A9587` | `#5C6270` | Secondary labels, captions, metadata |
| `--border` | `#DDE9E3` | `#2A2D36` | Hairline card borders and dividers |
| `--map-bg` | `#DFF0E8` | `#0D0F14` | Map container loading surface |

### 3.2 Domain & Semantic Accents

| Domain / State | Primary Token | Darker Tone | Light Tint | Extra Light Tint |
|---|---|---|---|---|
| **Agriculture / Vegetation** | `--accent-green: #35C778` | `--accent-green-d: #1FA85B` | `--accent-green-l: #C8F5DA` | `--accent-green-xl: #E8FBF0` |
| **Utility / Infrastructure** | `--accent-blue: #3AAFFF` | `--accent-blue-d: #1A8FE0` | `--accent-blue-l: #D0EEFF` | `--accent-blue-xl: #EAF6FF` |
| **Critical Alerts / Emergencies** | `--accent-red: #FF6B6B` | `--accent-red-d: #D94F4F` | `--accent-red-l: #FFE0E0` | `--accent-red-xl: rgba(255,107,107,0.12)` |
| **Warnings / Cautions** | `--accent-amber: #FFB347` | `--accent-amber-d: #A56A00` | `--accent-amber-l: #FFF0D0` | `--accent-amber-xl: rgba(255,179,71,0.12)` |

### 3.3 Dynamic Domain Switching Logic
```javascript
// Updates theme variables dynamically based on active domain selection
function applyDomainAccent(domain) {
  const map = {
    agri:    { c: 'var(--accent-green)', d: 'var(--accent-green-d)', xl: 'var(--accent-green-xl)' },
    utility: { c: 'var(--accent-blue)',  d: 'var(--accent-blue-d)',  xl: 'var(--accent-blue-xl)' },
    disaster:{ c: 'var(--accent-red)',   d: 'var(--accent-red-d)',   xl: 'var(--accent-red-xl)' }
  };
  const a = map[domain];
  document.documentElement.style.setProperty('--accent-domain', a.c);
  document.documentElement.style.setProperty('--accent-domain-d', a.d);
  document.documentElement.style.setProperty('--accent-domain-xl', a.xl);
}
```

---

## 3. Typography & Spacing System

**Font Stack**: `"Mitr"`, `"Kanit"`, `ui-sans-serif`, `system-ui`, `sans-serif`  
**Google Fonts Import**: `Mitr:wght@300;400;500;600` + `Kanit:ital,wght@0,300;0,400;0,500;0,600;1,400`

### 3.1 Typography Scale
| Element | Font Size | Weight | Line Height | Notes |
|---|---|---|---|---|
| Page Title | `2.0rem` (32px) | 600 | 1.15 | Main scenario title |
| Topbar Eyebrow | `0.72rem` (11.5px) | 500 | 1.0 | Uppercase with `letter-spacing: 0.08em` |
| Card Title | `0.90rem` (14.5px) | 600 | 1.2 | Standard card headline |
| KPI Metric Value | `2.40rem` (38.5px) | 700 | 1.0 | Primary numeric indicator |
| Body / Descriptions | `0.85–0.875rem` (14px) | 400 | 1.5 | General copy and details |
| Captions & Badges | `0.65–0.78rem` (11–12.5px) | 500–600 | 1.3 | Metadata, status chips, and axis tags |

### 3.2 Spacing, Radii & Shadows
- **Border Radii**: `--r-sm: 14px` · `--r-md: 20px` · `--r-lg: 28px` · `--r-xl: 36px` · `--r-pill: 9999px`
- **Elevations (Diffused Ambient Depth)**:
  - `--shadow-sm`: `0 2px 8px rgba(0,0,0,0.06)`
  - `--shadow-md`: `0 6px 24px rgba(0,0,0,0.09)`
  - `--shadow-lg`: `0 16px 48px rgba(0,0,0,0.13)`
- **Animation Timing**:
  - Fast Transitions: `--t-fast: 0.16s ease`
  - Spring Transitions: `--t-med: 0.38s cubic-bezier(0.34, 1.56, 0.64, 1)`

---

## 4. Modular Domain & Scenario Content Architecture

The application dynamically updates views across three main operational domains and nine modular scenarios:

```
├── Agriculture (Theme: Green Accent)
│   ├── Crop Inventory & Parcel Monitoring
│   ├── Crop Health Monitoring (NDVI / NDWI Indexes)
│   └── Anomaly & Early Warning (Change Detection)
│
├── Utility & Infrastructure (Theme: Blue Accent)
│   ├── Solar Farm Monitoring (Telemetry & Thermal Audits)
│   ├── Road Network Monitoring (Pavement Degradation)
│   └── Urban Expansion Tracking (Built-up Area Growth)
│
└── Disaster & Risk Management (Theme: Red / Amber Accent)
    ├── Flood & Drought Monitoring (Radar Inundation & SPI)
    ├── Landslide Risk Monitoring (Slope Stability & InSAR)
    └── Earthquake Impact Monitoring (Structural Damage & Access Routes)
```

---

## 5. Layout Architecture — 12-Column Asymmetric Bento Grid

- **Sidebar**: Default `248px` width with domain navigation buttons, sub-scenario lists, and dark mode toggle.
- **Main Viewport**: `margin-left: 248px`, `padding: 32px` (`18px` on mobile screens).
- **12-Column Grid Composition**:
  - **Tier 1 (Top KPIs)**: 3 × `col-4` metric cards with dynamic domain top borders and status badges.
  - **Tier 2 (Hero Interactive Area)**:
    - `col-8`: **Spatial Canvas** (Deck.GL map container with floating layer controls and legend).
    - `col-4`: **Live Alert Feed** (Severity-coded alert rows with timestamps and category chips).
  - **Tier 3 (Analytics & Feature Bento)**:
    - `col-4`: **Capability Card** (Feature checklist with domain-accented checkmarks).
    - `col-4`: **Bar Chart Card** (CSS bar chart with automatic height normalization).
    - `col-4`: **Regional Progress Card** (Progress tracks showing regional coverage percentages).
- **Supported Grid Column Classes**: `col-3`, `col-4`, `col-6`, `col-8`, `col-12`. All must be defined in CSS as `grid-column: span N;`.
- **Responsive Breakpoints**:
  - `≤ 1440px`: Sidebar collapses to icon-only mode (`64px` width).
  - `≤ 1100px`: Sidebar collapses off-canvas; grid transforms to `col-6` and `col-12`.
  - `≤ 600px`: Single-column vertical stack (`col-12` for all cards).

---

## 6. Interactive Canvas Engine (Deck.GL v9 + MapLibre GL)

### 6.1 Setup & Offline-Safe Tile Configuration
MapLibre GL serves as the raster basemap container, using an inline style object to avoid CORS issues in local `file://` environments:

```javascript
const BASEMAPS = {
  light: [
    'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png' // Fallback
  ],
  dark: [
    'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
    'https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png' // Fallback
  ]
};

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: { 'basemap': { type: 'raster', tiles: BASEMAPS[savedTheme], tileSize: 256 } },
    layers: [{ id: 'basemap', type: 'raster', source: 'basemap' }]
  },
  center: [100.523, 13.736],
  zoom: 10,
  attributionControl: false
});
```

### 6.2 Deck.GL Layer Stack & Controls
Integrated via `deck.MapboxOverlay` with dynamic checkbox layer toggles:

| Layer ID | Layer Type | Purpose & Aesthetics |
|---|---|---|
| `primary-layer` | `ScatterplotLayer` | Parcel/asset centroids styled with dynamic domain RGB (`rgba(r,g,b,130)`) and variable radius. |
| `secondary-layer`| `LineLayer` | Network interconnects / waterway paths (`[58, 175, 255, 180]`, `getWidth: 2`). |
| `alert-halo` | `ScatterplotLayer` | Low-opacity pulsing halo ring (`radius: 340`, `rgba(severity, 40)`). |
| `alert-dot` | `ScatterplotLayer` | Core alert marker (`radius: 110`, `rgba(severity, 230)`). |

### 6.3 Tooltip & UI Controls
- **Floating Controls (`.map-controls`)**: Positioned at `top: 16px; left: 16px;` with frosted card styling.
- **Floating Legend (`.map-legend`)**: Positioned at `bottom: 16px; right: 16px;`.
- **Custom Tooltip (`#deck-tooltip`)**: Positioned dynamically on hover with high-contrast typography and dark backdrop.

---

## 7. Component Reference & Styling Guidelines

### 7.1 KPI Card (`.kpi-card`)
- `border-top: 3px solid var(--accent-domain)`
- Circular icon/indicator container with `--accent-domain-xl` background
- Floating badge in top-right corner (`.up`, `.down`, `.warn`)

### 7.2 Alert Feed & List Rows (`.alert-row` / `.feed-item`)
- **Open & Borderless Structure**: Rows sit directly on the card canvas without enclosed pill frames or nested background bubbles (`background: transparent`).
- **Hairline Dividers**: Separated solely by a subtle bottom divider (`1px solid var(--border)`), with the last item borderless (`:last-child { border-bottom: none; }`).
- **Visual Hierarchy**:
  - Left indicator: Soft glowing or solid severity dot (`.high` Red `#FF6B6B`, `.mid` Amber `#FFB347`, `.low` Blue `#3AAFFF`).
  - Text stack: Crisp bold headline on top, muted subtitle metadata directly underneath.
  - Generous vertical padding (`12px–14px`) for optimal readability and airiness.

### 7.3 Domain Capability Card (`.domain-card`)
- Pill tag (`.domain-pill`) with `--accent-domain-xl` background
- Checkmark list (`.feature-check`) using `::before { content: "✓"; }`

### 7.4 Dynamic Bar Chart (`.bar-chart`)
- Height dynamically calculated: `(value / maxValue) * 100` px
- Rounded top corners: `border-radius: 8px 8px 3px 3px`
- Background styled with `var(--accent-domain)` and transition on state change

---

## 8. Theme System (Dark / Light Mode)

Theme is managed by the `data-theme` attribute on the `<html>` root:
- Default: `light` (or loaded from `localStorage.getItem('gislab-theme')`).
- **Dark Mode Color Standards**: Neutral charcoal/slate (`--bg: #131418`, `--bg-card: #1C1E24`). Backgrounds must **never** be tinted green in dark mode.
- Basemap swap executes via: `map.getSource('basemap').setTiles([BASEMAPS[mode]])`.

### 8.1 Special Interaction: Chromatic Aura Ripple (Theme Toggle Only)
To give theme and UI mode switching a distinct tactile feedback, interactive theme switchers feature a specialized **Chromatic Aura Ripple** effect on click:
- **Motion & Ripple**: A springy water-wave pulsation (`transform: scale(0.96) ➔ scale(1.03) ➔ scale(1)`).
- **Chromatic Aura**: A brief, subtle rainbow prism aura glowing softly around the button perimeter (`box-shadow` combining green, cyan, amber, and coral hues at low opacity).
- **Strict Usage Constraint**: This effect is **strictly reserved** for controls that transform the global UI state or color theme (e.g., Dark/Light Mode switch). It must **never** be applied to standard action buttons, forms, or navigation links.
- **Progressive Enhancement**: Wrap with `@media (prefers-reduced-motion: no-preference)` so it gracefully degrades on low-end devices or for users with motion sensitivity.

---

## 9. UI States (Loading, Empty, Error)

| State | Visual Treatment | Notes |
|---|---|---|
| **Loading** | Skeleton shimmer cards with `background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-hover) 50%, var(--bg-card) 75%); animation: shimmer 1.5s infinite;` | Show while API/tile data is fetching |
| **Empty** | Centered muted icon + text (e.g., "No data for selected period") inside the card container | Avoid blank cards |
| **Error** | Red-tinted banner (`--accent-red-l` background, `--accent-red` text/icon) with retry action | Display specific error context |

---

## 10. Accessibility Standards

- **Contrast Ratio**: Minimum WCAG AA compliance (4.5:1 for normal text, 3:1 for large text). All `--text-h` / `--text-body` on `--bg` and `--bg-card` surfaces must meet this.
- **Focus Indicators**: All interactive elements must show a visible focus ring (`outline: 2px solid var(--accent-domain); outline-offset: 2px;`) for keyboard navigation.
- **ARIA Labels**: All map controls, sidebar toggles, and interactive canvas buttons must include `aria-label` attributes.
- **Color-Blind Safety**: Do not rely on color alone to convey meaning. Pair colors with icons, patterns, or text labels (e.g., severity dots should include shape or text indicators alongside the color).

---

## 11. Emoji & Icon Policy

- **Default: No emoji.** Prefer SVG line icons or Unicode symbols for UI elements.
- If emoji is used for prototyping convenience, add an inline comment: `<!-- TODO: Replace emoji with SVG icon -->` to flag it for future cleanup.
- Production builds should replace all emoji with properly sized, accessible SVG icons.

---

## 12. Best Practices & Guidelines

**Do:**
- Rely on CSS variables (`var(--accent-domain)`) for all domain-specific styling.
- Keep canvas toolbars and legends floating over the visualization rather than wrapping with desktop software frames.
- Use deterministic random generators (`seededRandom`) for demo data points to ensure stable visual states.
- Maintain soft ambient drop shadows and large corner radii (`28px–36px`) for clean Bento cards.
- Provide tile source fallbacks for basemap reliability.

**Don't:**
- Use hard, dark drop shadows or heavy table outlines.
- Load external `style.json` MapLibre URLs without fallback (causes CORS failure on local `file://`).
- Use muddy green backgrounds in dark mode.
- Clutter the visualization canvas with fixed top toolbars.
- Use emoji in production code without flagging for replacement.
