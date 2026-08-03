---
layout: default
title: Projects
permalink: /projects/
---

<section class="section projects-section">
  <div class="projects-hero">
    <h2 class="section__title">Projects</h2>
    <p class="section__body">A curated selection of tools I've built and contributed to. Each one represents a distinct problem I care about solving.</p>
  </div>

  <div class="projects-grid">

    <!-- Project 1: litemind-ui -->
    <article class="project-card" data-card>
      <a href="https://github.com/debabratamishra/litemind-ui" target="_blank" rel="noopener noreferrer" class="project-card__link">
        <div class="project-card__image">
          <img src="{{ '/images/projects/litemind-ui-demo.gif' | relative_url }}" alt="litemind-ui demo screenshot" loading="lazy" width="800" height="450">
          <span class="project-card__play" aria-hidden="true">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="22" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.1)" stroke-width="1"/>
              <polygon points="19,14 34,24 19,34" fill="#2a2a2a"/>
            </svg>
          </span>
        </div>
        <div class="project-card__body">
          <h3 class="project-card__title">litemind-ui</h3>
          <p class="project-card__desc">A lightweight, accessible UI component library built on native Web Components. Zero runtime dependencies, themeable via CSS custom properties, and designed to drop into any project without a build step.</p>
          <div class="project-card__meta">
            <span class="project-card__tag">TypeScript</span>
            <span class="project-card__tag">Zero-dep</span>
          </div>
        </div>
      </a>
    </article>

    <!-- Project 2: litemind-cli -->
    <article class="project-card" data-card>
      <a href="https://github.com/debabratamishra/litemind-cli" target="_blank" rel="noopener noreferrer" class="project-card__link">
        <div class="project-card__image">
          <img src="{{ '/images/projects/litemind-cli-demo.png' | relative_url }}" alt="litemind-cli demo terminal screenshot" loading="lazy" width="800" height="450">
          <span class="project-card__play" aria-hidden="true">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="22" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.1)" stroke-width="1"/>
              <polygon points="19,14 34,24 19,34" fill="#2a2a2a"/>
            </svg>
          </span>
        </div>
        <div class="project-card__body">
          <h3 class="project-card__title">litemind-cli</h3>
          <p class="project-card__desc">A fast, battery-included CLI scaffolding tool that generates production-ready project structures with opinionated defaults. Supports plugins, custom templates, and interactive prompts.</p>
          <div class="project-card__meta">
            <span class="project-card__tag">CLI</span>
            <span class="project-card__tag">Templating</span>
          </div>
        </div>
      </a>
    </article>

    <!-- Project 3: llm-evals -->
    <article class="project-card" data-card>
      <a href="https://github.com/debabratamishra/llm-evals" target="_blank" rel="noopener noreferrer" class="project-card__link">
        <div class="project-card__image">
          <img src="{{ '/images/projects/llm-evals-demo.jpeg' | relative_url }}" alt="llm-evals dashboard demo screenshot" loading="lazy" width="800" height="450">
          <span class="project-card__play" aria-hidden="true">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="22" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.1)" stroke-width="1"/>
              <polygon points="19,14 34,24 19,34" fill="#2a2a2a"/>
            </svg>
          </span>
        </div>
        <div class="project-card__body">
          <h3 class="project-card__title">llm-evals</h3>
          <p class="project-card__desc">A framework for systematic evaluation of large language model outputs. Define rubrics, run parallel comparisons across models, and generate statistical reports with minimal configuration.</p>
          <div class="project-card__meta">
            <span class="project-card__tag">LLM Evaluation</span>
            <span class="project-card__tag">Performance Benchmarking</span>
          </div>
        </div>
      </a>
    </article>

  </div>

  <div class="projects-cta">
    <a href="https://github.com/debabratamishra" class="hero__cta-btn">View All on GitHub →</a>
  </div>
</section>