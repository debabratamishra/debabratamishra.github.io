# Site Redesign Design Spec

**Date:** 2026-07-30  
**Status:** Approved  
**Subject:** Debabrata Mishra personal portfolio/blog (Jekyll + Minimal Mistakes vendored theme)  
**Goal:** Upgrade from academic/minimalist to professional-grade studio/portfolio

---

## 1. Context

### Subject
AI/ML practitioner and Senior Data Scientist based in Sydney, publishing deep technical articles on multi-agent systems, graph engineering, LLM infrastructure, and related topics.

### Audience
Technically literate peers, AI/ML professionals, and researchers. The homepage must establish authority in the first 5 seconds.

### Single Job of the Homepage
Establish authority and draw visitors into the work.

## 2. Design Direction: Curated Dashboard

A dashboard studio layout where the visual design is purposeful, the typography carries personality, and the dual-accent color system encodes information rather than decorating.

---

## 3. Typography

### Display Face: **Space Grotesk** (variable)
- Geometric sans with distinctive character, weights Light through Bold
- Used for: headings, hero statement, all display-scale type
- Single variable font request (space-grotesk variable wght)

### Accent/Mono Face: **JetBrains Mono** (variable)
- Designed for screen readability with personality
- Used structurally for: labels, metadata, tags, inline code, numeric data, read-time estimates
- Reinforces the "dashboard/instrument" feel

### Type Scale
| Role | Size | Weight |
|------|------|--------|
| Hero name | 5–6rem | Bold (700) |
| Hero role | 1.25rem | Regular (400) |
| Hero thesis (mono) | 1rem | Regular (400), JetBrains Mono |
| Post title | 1.5rem | SemiBold (600) |
| Post meta | 0.875rem | Regular (400), JetBrains Mono |
| Body text | 1rem | Regular (400) |
| Small/caption | 0.75rem | Regular (400), JetBrains Mono |

---

## 4. Color Palette

Six tokens with light/dark themes:

| Token | Light (Default) | Dark |
|---|---|---|
| `--bg` | `#FAFAFA` | `#0F0F0F` |
| `--surface` | `#FFFFFF` | `#1A1A1A` |
| `--text` | `#1A1A1A` | `#E0E0E0` |
| `--accent` (primary) | `#0D9488` (teal) | `#2DD4BF` |
| `--accent-secondary` | `#D97706` (amber) | `#F59E0B` |
| `--muted` | `#6B6B6B` | `#8A8A8A` |

**Semantic roles:**
- `--accent`: links, interactive states, CTA buttons — signals "clickable"
- `--accent-secondary`: tags, category badges, metadata — signals "information"
- `--muted`: secondary text, captions, timestamps

The dual accent system has a structural purpose: teal handles interaction, amber handles content classification. This avoids the "decorative accent" trap.

---

## 5. Layout Concept (ASCII Wireframe)

```
┌──────────────────────────────────────────────────┐
│  HERO                                            │
│                                                  │
│  DEBABRATA MISHRA  —  Senior Data Scientist     │
│  Building systems that think in graphs           │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  [interactive SVG: graph network, hover   │  │
│  │   on nodes → expand]                        │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
├──────────────────────────────────────────────────┤
│  POSTS — structured 2-column grid                │
│                                                  │
│  ┌─────────────────┬────────────────────────────┐│
│  │  CARD 1          │  CARD 2                    ││
│  │  Graph Engineering│  Speech-to-Speech         ││
│  │  2026-07-26     │  2026-04-11                ││
│  │  [AI] [ML]      │  [AI] [NLP]               ││
│  ├─────────────────┼────────────────────────────┤│
│  │  CARD 3          │  CARD 4                    ││
│  └─────────────────┴────────────────────────────┘│
│                                                  │
├──────────────────────────────────────────────────┤
│  FOOTER                                          │
│  © 2026 Debabrata Mishra    GitHub · LinkedIn   │
└──────────────────────────────────────────────────┘
```

### Hero Section
- Left-aligned, mono type for the name
- Role line in Space Grotesk Regular
- Thesis sentence in JetBrains Mono at 1rem
- SVG network diagram in the upper-right corner, CSS-only hover transitions (nodes expand on hover, no animation library)
- No navigation chrome above the fold — the content is the design

### Posts Grid
- Two-column CSS Grid, responsive to single column on mobile
- Cards have: title (Space Grotesk SemiBold), date + read time (JetBrains Mono, `--muted`), category tags (JetBrains Mono, `--accent-secondary` background)
- No excerpt text — title + metadata carries the signal; clicking through reads the work

### Footer
- Mono text, social links right-aligned, copyright left-aligned
- Minimal separation from content via a thin `--muted` rule

---

## 6. Signature Element

The **typographic hero statement** — the site opens with the name and thesis in large mono type against `--bg`. This is the one memorable thing.

The interactive SVG network diagram in the hero corner provides visual life without competing with the type. It signals "this person works with graphs/networks" without being a literal illustration of the content.

---

## 7. Theme Toggle (Existing, Enhanced)

The existing light/dark toggle is extended to adjust accent brightness per mode:
- Light mode: teal (`--accent`) + amber (`--accent-secondary`)
- Dark mode: teal brightens to `#2DD4BF`, amber brightens to `#F59E0B` — contrast is tuned per mode, not roles swapped. Both accent roles remain consistent (teal = interaction, amber = metadata).

---

## 8. What's NOT Changing

- The vendored Minimal Mistakes theme structure is preserved
- All existing posts, pages, and data files remain in place
- Jekyll build process unchanged
- Existing comment system (Staticman) unchanged
- Analytics placeholder unchanged

---

## 9. Files to Create/Modify

| File | Action |
|------|--------|
| `_sass/_variables.scss` | Replace palette tokens, update type scale, add Space Grotesk/ JetBrains Mono font-face declarations |
| `_layouts/default.html` | Remove sidebar, adjust hero structure |
| `_layouts/single.html` | Adjust post card metadata presentation |
| `_layouts/archive.html` | Adjust archive to two-column grid |
| `_includes/masthead.html` | Redesign navigation — minimal, mono type |
| `_includes/footer.html` | Redesign footer — mono, minimal |
| `assets/css/main.scss` | Add imports for Space Grotesk, JetBrains Mono |
| `_sass/_theme-toggle.scss` | Extend dual-accent dark mode logic |
| `_sass/_masthead.scss` | Redesign masthead styles |
| `_sass/_page.scss` | Adjust for dashboard layout |
| `_sass/_archive.scss` | Adjust for two-column card grid |
| `_sass/_buttons.scss` | Add accent-styled buttons |
| `_data/navigation.yml` | Add social links |
| `_pages/about.md` | Update author bio presentation |
| `index.html` | Create homepage with dashboard layout |
| `assets/js/main.js` | Add SVG interactivity for hero network diagram |

---

## 10. Design Tokens Summary

- Space Grotesk Variable (display + body)
- JetBrains Mono Variable (mono accents)
- 6 color tokens, light/dark variants
- No border-radius reset (keep existing 4px)
- No new shadows (keep existing subtle box-shadow)
- Grid system: CSS Grid (not Susy) for posts layout
