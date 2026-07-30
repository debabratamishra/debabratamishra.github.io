# Site Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade debabratamishra.github.io from academic/minimal to professional-grade studio/portfolio with curated dashboard layout, Space Grotesk + JetBrains Mono typography, dual-accent color system (teal + amber), and light/dark themes.

**Architecture:** Incremental modification of the vendored Minimal Mistakes Jekyll theme. Replace design tokens in `_variables.scss`, redesign masthead/footer navigation, create a dashboard homepage with CSS Grid posts grid, extend the theme toggle for dual-accent dark mode, and add a hero SVG network diagram. All changes preserve existing posts, pages, and data files.

**Tech Stack:** Jekyll, Liquid templates, SCSS, vanilla JS (no frameworks, no animation libraries).

## Global Constraints

- Vendored Minimal Mistakes theme structure preserved (no gem updates, no external dependencies)
- All existing posts, pages, and data files remain in place
- Jekyll build process unchanged
- Existing comment system (Staticman) unchanged
- Analytics placeholder unchanged
- Fonts loaded via `@font-face` declarations (no npm packages, no CDN for fonts — self-host the woff2 files)
- Dark mode uses `[data-theme="dark"]` attribute selector (existing pattern)
- Breakpoints remain: `$small: 600px`, `$medium: 768px`, `$medium-wide: 900px`, `$large: 925px`, `$x-large: 1280px`
- Grid for dashboard posts: CSS Grid (not Susy)
- All accent colors defined as CSS custom properties (`var(--accent)`, `var(--accent-secondary)`) for runtime theming
- SVG hero diagram uses CSS-only hover transitions (no animation library, no JS animation)
- Remove sidebar from all page layouts (dashboard layout is full-width)

---

### Task 1: Add Space Grotesk and JetBrains Mono fonts

**Files:**
- Create: `assets/fonts/space-grotesk/SpaceGrotesk[wght].woff2`
- Create: `assets/fonts/jetbrains-mono/JetBrainsMono[wght].woff2`
- Modify: `_sass/_variables.scss` (Typography section)
- Modify: `assets/css/main.scss`

- [ ] **Step 1: Download and place font files**
  Download Space Grotesk wght variable font from Google Fonts and save to `assets/fonts/space-grotesk/SpaceGrotesk[wght].woff2` (woff2 format, smaller than ttf, universally supported)

- [ ] **Step 2: Download and place JetBrains Mono font file**
  Download JetBrains Mono wght variable font from Google Fonts and save to `assets/fonts/jetbrains-mono/JetBrainsMono[wght].woff2`

- [ ] **Step 3: Update `_sass/_variables.scss` typography section**
  Replace the typography block (from `$doc-font-size` through `$global-font-family`) with:
  ```scss
  $doc-font-size              : 16;
  $paragraph-indent           : false;
  $indent-var                 : 1.3em;

  $font-display               : 'Space Grotesk', sans-serif;
  $font-mono                  : 'JetBrains Mono', monospace;

  $global-font-family         : $font-display;
  $header-font-family         : $font-display;
  $caption-font-family        : $font-mono;
  ```

- [ ] **Step 4: Add @font-face declarations to `assets/css/main.scss`**
  Insert two `@font-face` blocks at the top of the file (before all `@import` lines):
  ```scss
  @font-face {
    font-family: 'Space Grotesk';
    font-style: normal;
    font-weight: 100 900;
    font-display: swap;
    src: url('/assets/fonts/space-grotesk/SpaceGrotesk[wght].woff2') format('woff2');
  }

  @font-face {
    font-family: 'JetBrains Mono';
    font-style: normal;
    font-weight: 100 900;
    font-display: swap;
    src: url('/assets/fonts/jetbrains-mono/JetBrainsMono[wght].woff2') format('woff2');
  }
  ```

- [ ] **Step 5: Commit**
  ```bash
  git add assets/fonts/ _sass/_variables.scss assets/css/main.scss
  git commit -m "feat: add Space Grotesk and JetBrains Mono variable fonts"
  ```

---

### Task 2: Replace color palette with dual-accent six-token system

**Files:**
- Modify: `_sass/_variables.scss` (Colors section)

**Depends on:** Task 1 (fonts are needed before palette tokens are consumed)

- [ ] **Step 1: Replace Colors section in `_sass/_variables.scss`**
  Find the Colors comment header and replace everything from there through the end of the color variable declarations (through `$link-color-visited`) with the new token system:
  ```scss
  /* ==========================================================================
   Color Tokens
   ========================================================================== */

  :root {
    --bg: #FAFAFA;
    --surface: #FFFFFF;
    --text: #1A1A1A;
    --accent: #0D9488;
    --accent-secondary: #D97706;
    --muted: #6B6B6B;
    --border-color: #DCDCDC;
    --link: #0D9488;
    --link-hover: #0A6B62;
  }

  [data-theme="dark"] {
    --bg: #0F0F0F;
    --surface: #1A1A1A;
    --text: #E0E0E0;
    --accent: #2DD4BF;
    --accent-secondary: #F59E0B;
    --muted: #8A8A8A;
    --border-color: #2A2A2A;
    --link: #2DD4BF;
    --link-hover: #5EEAD4;
  }

  /* Legacy SCSS aliases for backward compat */
  $gray: var(--muted);
  $dark-gray: var(--text);
  $lighter-gray: var(--border-color);
  $light-gray: var(--surface);
  $body-color: var(--surface);
  $background-color: var(--bg);
  $text-color: var(--text);
  $border-color: var(--border-color);
  $primary-color: var(--accent);
  $link-color: var(--accent);
  $link-color-hover: var(--link-hover);
  $link-color-visited: var(--muted);
  $masthead-link-color: var(--accent);
  $masthead-link-color-hover: var(--link-hover);
  ```

  **Note:** Some SCSS functions (`mix()`, `darken()`, etc.) cannot operate on `var()` expressions. Any component that used `mix(#000, $link-color, 25%)` with the old color token should be checked after this change. The theme toggle (`_theme-toggle.scss`) will be updated in Task 6 to handle this.

- [ ] **Step 2: Commit**
  ```bash
  git add _sass/_variables.scss
  git commit -m "feat: replace color palette with dual-accent six-token CSS custom property system"
  ```

---

### Task 3: Redesign masthead and navigation

**Files:**
- Modify: `_includes/masthead.html`
- Modify: `_sass/_masthead.scss`
- Modify: `_data/navigation.yml`

**Depends on:** Task 2 (uses `--accent`, `--muted`, `--text` CSS custom properties)

- [ ] **Step 1: Update `_data/navigation.yml`**
  ```yaml
  # main navigation links
  main:
    - title: "Blog"
      url: /year-archive/
    - title: "CV"
      url: /cv/
  ```

- [ ] **Step 2: Replace `_includes/masthead.html`**
  Replace with a minimal mono masthead — site title left, nav links right, theme toggle preserved:
  ```html
  {% include base_path %}

  <div class="masthead">
    <div class="masthead__inner-wrap">
      <div class="masthead__menu">
        <nav id="site-nav" class="greedy-nav">
          <ul class="visible-links">
            <li class="masthead__menu-item masthead__menu-item--logo">
              <a href="{{ base_path }}/">{{ site.title }}</a>
            </li>
            {% for link in site.data.navigation.main %}
              {% if link.url contains 'http' %}
                {% assign domain = '' %}
                {% else %}
                {% assign domain = base_path %}
              {% endif %}
              <li class="masthead__menu-item"><a href="{{ domain }}{{ link.url }}">{{ link.title }}</a></li>
            {% endfor %}
          </ul>
          <ul class="hidden-links hidden"></ul>
        </nav>
      </div>

      <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle light / dark theme" aria-pressed="false" title="Switch to dark theme">
        <svg class="theme-toggle__icon theme-toggle__icon--sun" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="4"></line>
          <line x1="12" y1="20" x2="12" y2="23"></line>
          <line x1="1" y1="12" x2="4" y2="12"></line>
          <line x1="20" y1="12" x2="23" y2="12"></line>
          <line x1="4.2" y1="4.2" x2="6.3" y2="6.3"></line>
          <line x1="17.7" y1="17.7" x2="19.8" y2="19.8"></line>
          <line x1="4.2" y1="19.8" x2="6.3" y2="17.7"></line>
          <line x1="17.7" y1="6.3" x2="19.8" y2="4.2"></line>
        </svg>
        <svg class="theme-toggle__icon theme-toggle__icon--moon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>
      </button>
    </div>
  </div>
  ```

- [ ] **Step 3: Replace `_sass/_masthead.scss`**
  Replace with minimal dark mono styling:
  ```scss
  .masthead {
    position: relative;
    border-bottom: 1px solid var(--border-color);
    background: var(--surface);
    z-index: 20;
    padding: 0.75em 0;
    -webkit-animation: intro 0.3s both;
            animation: intro 0.3s both;
    -webkit-animation-delay: 0.15s;
            animation-delay: 0.15s;

    &__inner-wrap {
      max-width: 1280px;
      margin: 0 auto;
      padding: 0 1.5em;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: $font-display;
    }

    &__menu {
      ul {
        margin: 0;
        padding: 0;
        list-style-type: none;
        display: flex;
        gap: 2em;
      }
    }

    .masthead__menu-item {
      font-family: $font-mono;
      font-size: 0.8rem;
      letter-spacing: 0.02em;

      &--logo {
        font-family: $font-display;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: -0.01em;
      }

      a {
        color: var(--text);
        text-decoration: none;
        transition: color 0.2s ease;

        &:hover {
          color: var(--accent);
        }
      }
    }

    .greedy-nav .hidden-links {
      display: none;
    }
  }
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add _includes/masthead.html _sass/_masthead.scss _data/navigation.yml
  git commit -m "feat: redesign masthead as minimal mono navigation"
  ```

---

### Task 4: Create dashboard homepage

**Files:**
- Create: `index.html`
- Modify: `_layouts/single.html` (remove sidebar include)
- Modify: `_sass/_page.scss` (add dashboard styles)

**Depends on:** Tasks 1–3 (fonts, palette, masthead)

- [ ] **Step 1: Create `index.html`**
  New homepage with dashboard layout — hero (typographic statement + network SVG) + posts grid:
  ```html
  ---
  layout: default
  title: Debabrata Mishra
  ---

  <section class="hero">
    <div class="hero__content">
      <h1 class="hero__name">{{ site.title | markdownify }}</h1>
      <p class="hero__role">{{ site.author.bio }}</p>
      <p class="hero__thesis">Building systems that think in graphs</p>
    </div>
    <div class="hero__visual">
      <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Decorative network graph">
        <g class="hero__links">
          <line class="hero__link" x1="100" y1="40" x2="60" y2="80"/>
          <line class="hero__link" x1="100" y1="40" x2="140" y2="80"/>
          <line class="hero__link" x1="60" y1="80" x2="40" y2="140"/>
          <line class="hero__link" x1="60" y1="80" x2="80" y2="130"/>
          <line class="hero__link" x1="140" y1="80" x2="160" y2="140"/>
          <line class="hero__link" x1="140" y1="80" x2="120" y2="130"/>
          <line class="hero__link" x1="40" y1="140" x2="80" y2="130"/>
          <line class="hero__link" x1="160" y1="140" x2="120" y2="130"/>
          <line class="hero__link" x1="100" y1="40" x2="100" y2="0"/>
          <line class="hero__link" x1="100" y1="0" x2="60" y2="40"/>
          <line class="hero__link" x1="100" y1="0" x2="140" y2="40"/>
        </g>
        <g class="hero__nodes">
          <circle class="hero__node" cx="100" cy="0" r="4"/>
          <circle class="hero__node" cx="60" cy="40" r="4"/>
          <circle class="hero__node" cx="140" cy="40" r="4"/>
          <circle class="hero__node" cx="100" cy="40" r="4"/>
          <circle class="hero__node" cx="40" cy="140" r="4"/>
          <circle class="hero__node" cx="80" cy="130" r="4"/>
          <circle class="hero__node" cx="120" cy="130" r="4"/>
          <circle class="hero__node" cx="160" cy="140" r="4"/>
        </g>
      </svg>
    </div>
  </section>

  <section class="posts-dashboard">
    <h2 class="posts-dashboard__title">Recent Work</h2>
    <div class="posts-dashboard__grid">
      {% for post in site.posts limit:6 %}
        <article class="post-card">
          <h3 class="post-card__title">
            <a href="{{ post.url | relative_url }}">{{ post.title | markdownify | strip_html | strip_newlines }}</a>
          </h3>
          <time class="post-card__date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %d, %Y" }}</time>
          <div class="post-card__tags">
            {% for tag in post.tags limit:2 %}
              <span class="post-card__tag">{{ tag }}</span>
            {% endfor %}
          </div>
        </article>
      {% endfor %}
    </div>
    <a href="{{ base_path }}/year-archive/" class="btn btn--accent">All posts</a>
  </section>
  ```
  The `strip_html | strip_newlines` filters protect against Markdown heading markup appearing in the title.

- [ ] **Step 2: Update `_layouts/single.html`** — remove sidebar include
  Find the line `{% include sidebar.html %}` inside `<div id="main" ...>` and remove it. The article should now take the full width of the container.

- [ ] **Step 3: Add dashboard styles to `_sass/_page.scss`**
  Append the following at the end of the file:
  ```scss
  /* ==========================================================================
   HOME — Dashboard Layout
   ========================================================================== */

  .hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 4em 0 2em;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 3em;

    @media (max-width: 600px) {
      flex-direction: column;
    }
  }

  .hero__content {
    flex: 1;
    max-width: 600px;
  }

  .hero__name {
    font-family: var(--font-display, #{$font-display});
    font-size: clamp(2.5rem, 6vw, 5rem);
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
    color: var(--text);
  }

  .hero__role {
    font-family: $font-display;
    font-size: 1.25rem;
    font-weight: 400;
    color: var(--muted);
    margin: 0.5em 0 0;
  }

  .hero__thesis {
    font-family: $font-mono;
    font-size: 1rem;
    color: var(--accent);
    margin: 1em 0 0;
    padding: 0;
    border: none;
  }

  .hero__visual {
    flex-shrink: 0;
    margin-left: 2em;
    width: 160px;
    height: 160px;

    @media (max-width: 600px) {
      margin-left: 0;
      margin-top: 1.5em;
      width: 120px;
      height: 120px;
    }
  }

  /* SVG node/link hover — CSS only */
  .hero__node {
    fill: var(--accent);
    opacity: 0.7;
    transition: opacity 0.3s ease, r 0.3s ease;
  }

  .hero__link {
    stroke: var(--border-color);
    stroke-width: 0.5;
    opacity: 0.5;
    transition: opacity 0.3s ease, stroke 0.3s ease;
  }

  .hero__svg:hover .hero__node { opacity: 1; }
  .hero__svg:hover .hero__link { opacity: 1; stroke: var(--accent); }

  /* Posts Dashboard */
  .posts-dashboard {
    margin-top: 2em;
  }

  .posts-dashboard__title {
    font-family: $font-display;
    font-size: $type-size-4;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1.5em;
    padding-bottom: 0.5em;
    border-bottom: 1px solid var(--border-color);
  }

  .posts-dashboard__grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.5em;

    @media (min-width: 768px) {
      grid-template-columns: 1fr 1fr;
    }
  }

  .post-card {
    background: var(--surface);
    border: 1px solid var(--border-color);
    border-radius: $border-radius;
    padding: 1.5em;
    transition: border-color 0.2s ease;

    &:hover {
      border-color: var(--accent);
    }
  }

  .post-card__title {
    font-family: $font-display;
    font-size: $type-size-5;
    font-weight: 600;
    margin: 0 0 0.5em;
    line-height: 1.3;

    a {
      color: var(--text);
      text-decoration: none;

      &:hover { color: var(--accent); }
    }
  }

  .post-card__date {
    font-family: $font-mono;
    font-size: $type-size-7;
    color: var(--muted);
    display: block;
    margin-bottom: 0.75em;
  }

  .post-card__tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
  }

  .post-card__tag {
    font-family: $font-mono;
    font-size: $type-size-7;
    background: var(--accent-secondary);
    color: var(--surface);
    padding: 0.15em 0.5em;
    border-radius: 2px;
    opacity: 0.85;
  }

  .posts-dashboard .btn--accent {
    display: inline-block;
    margin-top: 2em;
  }
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add index.html _layouts/single.html _sass/_page.scss
  git commit -m "feat: create dashboard homepage with hero + posts grid"
  ```

---

### Task 5: Add accent button and card style overrides

**Files:**
- Modify: `_sass/_buttons.scss`

**Depends on:** Task 4 (dashboard calls `.btn--accent`)

- [ ] **Step 1: Add accent button variant to `_sass/_buttons.scss`**
  Append at the end:
  ```scss
  /* Accent button — uses theme primary accent */
  .btn--accent {
    display: inline-block;
    margin-bottom: 0.25em;
    padding: 0.5em 1.25em;
    font-family: $font-mono;
    font-size: $type-size-6;
    font-weight: bold;
    text-align: center;
    text-decoration: none;
    color: var(--surface) !important;
    background-color: var(--accent);
    border: 0 !important;
    border-radius: $border-radius;
    cursor: pointer;
    transition: opacity 0.2s ease;

    &:hover { opacity: 0.85; }
  }
  ```

- [ ] **Step 2: Commit**
  ```bash
  git add _sass/_buttons.scss
  git commit -m "feat: add accent button variant for dashboard CTA"
  ```

---

### Task 6: Redesign footer with mono, minimal layout

**Files:**
- Modify: `_includes/footer.html`
- Modify: `_sass/_footer.scss`

**Depends on:** Task 2 (uses `--muted` tokens)

- [ ] **Step 1: Replace `_includes/footer.html`**
  Replace with a mono-typed minimal footer:
  ```html
  {% include base_path %}

  <div class="page__footer-follow">
    <ul class="social-icons">
      {% if site.author.github %}
        <li><a href="http://github.com/{{ site.author.github }}"><i class="fab fa-github" aria-hidden="true"></i> GitHub</a></li>
      {% endif %}
      {% if site.author.linkedin %}
        <li><a href="http://linkedin.com/in/{{ site.author.linkedin }}"><i class="fab fa-linkedin" aria-hidden="true"></i> LinkedIn</a></li>
      {% endif %}
      {% if site.author.google %}
        <li><a href="{{ site.author.google }}"><i class="fab fa-google" aria-hidden="true"></i> Google Scholar</a></li>
      {% endif %}
    </ul>
  </div>

  <div class="page__footer-copyright">
    &copy; {{ site.time | date: '%Y' }} {{ site.name | default: site.title }}.
    Powered by <a href="http://jekyllrb.com" rel="nofollow">Jekyll</a> &amp; <a href="https://github.com/academicpages/academicpages.github.io">AcademicPages</a>.
  </div>
  ```

- [ ] **Step 2: Update `_sass/_footer.scss`** — replace with minimal mono footer
  ```scss
  .page__footer {
    @include clearfix;
    float: left;
    margin-left: 0;
    margin-right: 0;
    width: 100%;
    clear: both;
    position: absolute;
    bottom: 0em;
    height: auto;
    margin-top: 3em;
    color: var(--muted, #{$gray});
    background-color: var(--surface);
    border-top: 1px solid var(--border-color);

    @media (prefers-color-scheme: dark) {
      background-color: #1A1A1A;
    }
  }

  .page__footer-copyright {
    font-family: $font-mono;
    font-size: $type-size-7;
  }

  .page__footer-follow {
    ul {
      margin: 0;
      padding: 0;
      list-style-type: none;
      display: flex;
      gap: 1em;
    }

    li {
      display: inline-block;
      font-family: $font-mono;
      font-size: $type-size-6;
    }

    a {
      color: var(--text, #{$text-color});
      text-decoration: none;
      font-weight: normal;

      &:hover {
        color: var(--accent, #{$primary-color});
        text-decoration: underline;
      }
    }

    .fab {
      margin-right: 0.25em;
    }
  }
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add _includes/footer.html _sass/_footer.scss
  git commit -m "feat: redesign footer as minimal mono layout"
  ```

---

### Task 7: Extend theme toggle for dual-accent dark mode

**Files:**
- Modify: `_sass/_theme-toggle.scss`

**Depends on:** Task 2 (new palette tokens)

- [ ] **Step 1: Replace the dark palette in `_sass/_theme-toggle.scss`**
  Find the `[data-theme="dark"]` block and replace all CSS custom property declarations (from `--bg` through `--code-bg`) with the new dual-accent palette, matching the light default values defined in `_variables.scss`:
  ```scss
  [data-theme="dark"] {
    --bg: #0F0F0F;
    --surface: #1A1A1A;
    --text: #E0E0E0;
    --muted: #8A8A8A;
    --border: #2A2A2A;
    --link: #2DD4BF;
    --link-hover: #5EEAD4;
    --accent: #2DD4BF;
    --code-bg: #1a1d22;
  }
  ```
  **Note:** Remove the old `--accent: #8a929a` and the hardcoded `--border: #2c3138` / `--muted: #9aa4ad` lines. The new teal + amber values replace the old gray accent and all other dark palette tokens.

- [ ] **Step 2: Update remaining hardcoded dark-mode colors to use CSS custom properties**
  In the same `[data-theme="dark"]` block, replace hardcoded hex values that should now be theme-dependent:
  - Replace `#e7ecf1` headings color with `var(--text)`
  - Replace `#7fb7c9` visited links with `var(--link-hover)`
  - Replace `#23272e` hover backgrounds with `var(--surface)` shifted by 5% (use a computed value or keep as hardcoded since it's a subtle UI detail)
  - Replace `#002b36` / `#073642` code-block dark mode background (these are the Solarized Dark colors — keep them as-is since they're specific to the syntax theme, not the site palette)

- [ ] **Step 3: Update dark-mode theme toggle icon visibility rules**
  Make sure the sun/moon visibility rules match the new theme (no change needed — these are independent of palette):
  ```scss
  [data-theme="dark"] .theme-toggle__icon--sun  { display: block; }
  [data-theme="dark"] .theme-toggle__icon--moon { display: none; }
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add _sass/_theme-toggle.scss
  git commit -m "feat: extend theme toggle with dual-accent dark palette (teal + amber)"
  ```

---

### Task 8: Remove sidebar from layouts and clean up unused imports

**Files:**
- Modify: `_layouts/default.html` (check for sidebar include)
- Modify: `_sass/_sidebar.scss` (mark as deprecated, don't delete)
- Modify: `assets/css/main.scss` (remove sidebar import if appropriate)

**Depends on:** Tasks 3–4 (masthead and homepage already don't use sidebar)

- [ ] **Step 1: Verify no layout includes sidebar**
  Grep for `{% include sidebar.html %}` across `_layouts/` to confirm it was already removed from `single.html` in Task 4. Check `_layouts/default.html`, `_layouts/archive.html`, `_layouts/archive-taxonomy.html` — if any include sidebar, remove the include.

- [ ] **Step 2: Check if Susy grid import is still needed**
  Grep for `@include gallery`, `@include span`, `@include prefix` across `_sass/`. If the remaining usage is only in `_archive.scss` and you want to modernize the archive grid too (optional, out of scope for this redesign), skip this step. Otherwise, keep the Susy import in `main.scss`.

- [ ] **Step 3: Commit**
  ```bash
  git add _layouts/ assets/css/main.scss _sass/_sidebar.scss
  git commit -m "chore: remove sidebar references from layouts, clean up imports"
  ```

---

### Task 9: Add SVG hero interactivity (CSS-only hover, minimal JS fallback)

**Files:**
- Modify: `assets/js/main.js`

**Depends on:** Task 4 (SVG exists in homepage)

- [ ] **Step 1: Add SVG node hover expansion**
  In `assets/js/main.js`, add a minimal JS enhancement for nodes in the hero SVG (optional — CSS hover already handles the visual effect). This JS step adds keyboard accessibility: pressing Enter on a focused node activates its hover state:
  ```js
  // Hero SVG network diagram — keyboard accessibility
  document.addEventListener('DOMContentLoaded', () => {
    const nodes = document.querySelectorAll('.hero__node');
    nodes.forEach(node => {
      node.setAttribute('tabindex', '0');
      node.setAttribute('role', 'button');
      node.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          node.parentElement.classList.toggle('hero__svg--active');
        }
      });
    });
  });
  ```
  The CSS `.hero__svg--active` rule (in `_page.scss`) toggles expanded node radius. This is optional polish — the CSS-only hover handles the primary interaction.

- [ ] **Step 2: Commit**
  ```bash
  git add assets/js/main.js
  git commit -m "feat: add keyboard accessibility for hero SVG network diagram"
  ```

---

### Task 10: Update about page and navigation data

**Files:**
- Modify: `_pages/about.md`
- Modify: `_config.yml` (social links)

**Depends on:** All previous tasks

- [ ] **Step 1: Update `_pages/about.md`** — tighten the bio to match the new dashboard tone. Remove verbose prose and keep the thesis statement format consistent with the homepage:
  ```markdown
  ---
  layout: single
  title: About
  permalink: /about/
  ---

  Hey there — I'm Deb, a Senior Data Scientist at Commonwealth Bank of Australia in Sydney. I build multi-agent AI systems and graph-based architectures for complex decision problems. My work spans efficient on-device AI, LLM infrastructure, and medical imaging diagnostics.

  <!-- end of concise about text -->
  ```
  The about page's role is to reinforce the homepage thesis, not repeat it. A short, confident bio with a research/engineering thesis.

- [ ] **Step 2: Update social links in `_config.yml`** — ensure `github` and `linkedin` are populated:
  ```yaml
  social:
    links:
      - name: GitHub
        url: https://github.com/debabratamishra
      - name: LinkedIn
        url: https://linkedin.com/in/debabrata-mishra1
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add _pages/about.md _config.yml
  git commit -m "chore: update about page bio and social links"
  ```

---

### Task 11: Responsive polish and cross-theme verification

**Files:**
- Review: `_sass/_page.scss`, `_sass/_masthead.scss`, `_sass/_footer.scss`
- Test: local Jekyll build (`bundle exec jekyll serve`)

**Depends on:** All other tasks

- [ ] **Step 1: Verify light theme renders correctly**
  Run `bundle exec jekyll serve` and visit `http://localhost:4000`. Confirm:
  - Hero typographic statement (name + role + thesis) renders in Space Grotesk / JetBrains Mono
  - Network SVG diagram displays and nodes highlight on hover
  - Posts grid shows two columns on desktop, single column on mobile
  - Tags display in amber (`--accent-secondary`)
  - "All posts" button renders in teal (`--accent`)
  - Masthead is minimal mono navigation
  - Footer is mono and minimal

- [ ] **Step 2: Verify dark theme renders correctly**
  Toggle theme via the theme switch in the masthead. Confirm:
  - All text remains readable against dark backgrounds
  - Teal links/buttons remain visible against dark surface
  - Amber tags remain visible in dark mode
  - SVG nodes and links adapt to dark theme (check contrast)
  - Theme toggle switch itself is visible and correctly labeled

- [ ] **Step 3: Verify mobile rendering**
  Resize browser to 375px width (mobile). Confirm:
  - Hero stacks vertically (name above SVG)
  - Posts grid is single column
  - Masthead navigation doesn't overflow
  - Theme toggle is accessible

- [ ] **Step 4: Verify reduced motion preference**
  In browser dev tools, enable `prefers-reduced-motion: reduce`. Confirm:
  - No CSS transitions or animations fire
  - SVG hover expansion is disabled
  
- [ ] **Step 5: Commit**
  ```bash
  git add -A
  git commit -m "chore: responsive polish, cross-theme verification"
  ```
