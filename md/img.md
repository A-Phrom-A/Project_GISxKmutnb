# GISlab True Miracle — Image & Prompt Generation System (`img.md`)

> **Purpose**: Standardized AI image generation guide (`generate_image`) designed to maintain brand identity, color tokens, layout hierarchy, and aesthetic consistency across all generated visuals, matching **`design-web.md`** and **`design-presentation.md`**.

---

## 1. User Intake & Clarification Framework

Before constructing an image generation prompt, clarify the following requirements (either by asking the user directly or determining from context):

###  Questionnaire Matrix

| Step | Clarifying Question | Options / Selection |
|---|---|---|
| **Q1. Primary Objective** | What is the image intended for? | • **Full Slide Mockup** (16:9 pitch deck presentation)<br>• **UI Dashboard Card** (Isolated widget/map view)<br>• **Infographic Graphic** (Concept/data workflow diagram)<br>• **Hero / Banner Asset** (Website header graphic) |
| **Q2. Background & Canvas** | Does it require a background or canvas frame? | • **Solid Canvas** (`#FFFFFF` pure white or `#F3F8F5` light tint)<br>• **Dark Canvas** (`#131418` neutral charcoal slate)<br>• **Transparent / No Background** (PNG with alpha channel for compositing)<br>• **Frameless / Isolated UI Card** (No outer device bezel/mockup) |
| **Q3. Domain & Palette** | Which business domain / color accent? | • ** Agriculture**: Mint Green (`#35C778`, `#C8F5DA`)<br>• ** Utility & Infra**: Sky Blue (`#3AAFFF`, `#D0EEFF`)<br>• ** Disaster & Crisis**: Neutral Dark Slate with Red (`#FF6B6B`) & Amber (`#FFB347`)<br>• ** General Overview**: Balanced Multi-accent |
| **Q4. Target Aspect Ratio** | What display ratio is required? | • **`16:9`** (Presentation slides, widescreen dashboard)<br>• **`4:3` / `3:2`** (Standard bento card / report illustration)<br>• **`1:1`** (Square component, badge, avatar, icon block) |

---

## 2. Core Aesthetic & Brand Guidelines

All generated images must follow these strict visual principles:

- **Style**: Modern SaaS Dashboard, Minimalist, Elevated, Clean Flat Vector with Soft 3D Depth.
- **Personality**: High-tech, approachable, playful yet professional (avoid rigid legacy GIS desktop software styles).
- **Layout Logic**: **Bento Box Grid** with generous corner radius (`20px–36px`) and diffused ambient drop shadows.
- **Typography Style**: Geometric rounded sans-serif (matching Mitr/Kanit), large crisp numerical KPIs, clear labels.

### Negative Constraints & Rules
- `no device frames` (No laptops, desktop monitors, tablets, or phone mockups).
- `no muddy or olive-green dark backgrounds` (Dark mode must strictly use neutral charcoal/slate `#131418`).
- `no illegible garbled AI text` (Focus on clean geometric iconography and crisp headlines).
- `no photorealistic mess or noisy textures` (Keep interfaces flat, vector-styled, and sleek).
- **Default Background**: If the background color is not explicitly specified in the request, the image MUST be generated with a **transparent background** (`transparent background`, `no background`).

---

## 3. Color Tokens & Theme Specifications

### Light Theme (Agri, Utility, Overview, General Demos)
- **App Canvas Background**: `#FFFFFF` (Pure white) or `#F3F8F5` (Soft tinted surface)
- **Card Surface**: `#FFFFFF` with soft diffused shadow (`box-shadow: 0 8px 30px rgba(0,0,0,0.06)`)
- **Card Borders**: Subtle hairline border `#DDE9E3`
- **Text & Headings**: `#1B2B22` (Deep slate heading), `#3D5247` (Body text)
- **Accents**:
  - Agriculture: Mint Green `#35C778` / Light Mint `#C8F5DA`
  - Utility / Water: Sky Blue `#3AAFFF` / Light Sky `#D0EEFF`
  - Warnings: Soft Amber `#FFB347` / Light Amber `#FFF0D0`
  - Critical Alerts: Coral Red `#FF6B6B` / Light Coral `#FFE0E0`

### Dark Theme (Disaster Risk, Emergency Alerts, Night Monitoring)
- **App Canvas Background**: Neutral Dark Slate `#131418` *(never green-tinted)*
- **Card Surface**: `#1C1E24` (Sleek charcoal card) with crisp boundary stroke `#2A2D36`
- **Text & Headings**: `#F0F2F5` (Crisp light heading), `#A8AEBA` (Muted body text)
- **Accents**: High-contrast glowing indicators (Coral Red `#FF6B6B`, Warning Amber `#FFB347`, Cyan `#3AAFFF`)

---

## 4. Prompt Engineering Formula

Compose every generation prompt using the following 5-layer recipe:

```
[Asset Type & Viewport] + [Core Subject & Data Elements] + [Layout & Bento Structure] + [Color Tokens & Theme] + [Rendering Rules & Negative Constraints]
```

### Essential Keywords:
- **Structure**: `modern SaaS dashboard UI mockup`, `bento grid layout`, `rounded card widgets (28px radius)`, `soft diffused drop shadows`, `shadcn UI aesthetic`, `minimalist flat vector graphics`
- **Typography**: `crisp modern geometric sans-serif text`, `large bold numerical metrics`, `clean minimalist data tags`
- **Quality & Constraints**: `user interface screenshot`, `no device frames`, `no laptop or phone frames`, `clean vector graphic`, `8k resolution UI concept`

---

## 5. Production-Ready Prompt Templates

### 1. Agriculture / Crop Health Deck (Light Theme)
```text
A premium minimalist web app demo mockup for an Agriculture dashboard, light theme. Dominant crisp white background (#FFFFFF) with soft mint-green accents (#35C778, #C8F5DA). Clean bento grid layout with highly rounded corners (28px) and soft diffused shadows. Displays a circular crop health monitoring gauge (0.74 NDVI), active parcel inventory cards, and a crisp satellite vegetation map widget. Modern geometric sans-serif typography, clean flat vector interface design, user interface screenshot, no device frames, 16:9 aspect ratio.
```

### 2. Utility & Renewable Energy Dashboard (Light Theme)
```text
A modern minimalist dashboard user interface for Utility Infrastructure monitoring, light theme. Dominant white background with soft sky-blue accents (#3AAFFF, #D0EEFF). Clean bento grid layout with rounded corners (28px). Displays a solar farm telemetry map widget with outlined panel clusters, road condition percentage gauge, and clean system alert cards. Minimalist vector styling, clear modern typography, user interface screenshot, no device frames, 16:9 aspect ratio.
```

### 3. Disaster Risk & Emergency Management (Dark Mode)
```text
A premium dark mode crisis dashboard user interface for Disaster Management. Neutral dark slate canvas (#131418, not green-tinted) with sleek charcoal cards (#1C1E24). Clean bento grid layout with rounded corners (28px). High-contrast glowing red flood inundation contour maps (#FF6B6B), amber warning indices for landslides (#FFB347), and earthquake severity status tags. Modern sans-serif text, clean flat UI design, user interface screenshot, no device frames, 16:9 aspect ratio.
```

### 4. Isolated Map Widget / Detail Component (Frameless)
```text
A clean geospatial map widget interface card, light theme. Shows clean vector parcel boundaries colored in NDVI gradient (light-green to deep forest green) with one red anomaly highlighted. Minimalist legend box in the corner, smooth map styling, white UI card container with rounded corners (28px) and soft shadow, user interface component, no device frames, 4:3 aspect ratio.
```

---

## 6. Non-Dashboard Prompt Templates

> These templates are designed to work across multiple AI image generation platforms including **DALL-E (GPT)**, **Claude (Anthropic generate_image)**, **Gemini Imagen**, **Midjourney**, and **Stable Diffusion**. Use clear, descriptive natural language and avoid platform-specific syntax.

### 5. Isometric 3D Illustration (Cross-Platform Compatible)
```text
A highly detailed 3D isometric illustration of a smart agricultural monitoring system. A cross-section view of farmland showing layers of brown soil with visible root systems and glowing blue digital irrigation pipes underneath. Small high-tech IoT sensors planted in the ground emit soft holographic data readouts. Lush green crops grow on the surface. A tiny futuristic drone hovers above scanning the field. Soft studio lighting, clay-render style, vibrant green and earthy brown colors, transparent background, high resolution, stylized and modern.
```

### 6. Satellite / Drone Realistic View (Cross-Platform Compatible)
```text
A realistic top-down satellite view of agricultural land parcels with clearly delineated field boundaries. Some fields show lush green healthy vegetation while others display dry brown patches indicating drought stress. A thin overlay of semi-transparent colored zones (green for healthy, amber for warning, red for critical) is applied. Clean cartographic style, natural earth tones, transparent background, high resolution, 16:9 aspect ratio.
```

### 7. Infographic / Data Flow Diagram (Cross-Platform Compatible)
```text
A clean, modern infographic-style data flow diagram showing a geospatial AI processing pipeline. On the left, satellite and drone icons feed raw data into a central AI processing node represented by a glowing circuit brain. On the right, output branches show a dashboard card, an alert notification, and a map widget. Soft mint-green and sky-blue color accents on a transparent background. Minimalist flat vector style, geometric sans-serif labels, clean connecting arrows, 3:2 aspect ratio.
```

---

## 7. Pre-Generation Checklist

- [ ] Clarified intended use (Presentation slide vs. isolated widget vs. infographic vs. 3D illustration).
- [ ] Confirmed background requirements (Solid white `#FFFFFF`, Dark slate `#131418`, Transparent, or standalone card).
- [ ] Enforced `rounded corners (20px–28px)` and `bento grid layout` (for dashboard prompts).
- [ ] Set appropriate aspect ratio (`16:9` for slides, `4:3` for cards, `1:1` for icons/badges, `3:2` for infographics).
- [ ] Verified prompt is platform-agnostic (no Midjourney `--v` flags or DALL-E-specific syntax).
