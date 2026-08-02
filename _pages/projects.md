---
layout: default
title: Projects
permalink: /projects/
---

<section class="section projects-section">
  <div class="projects-hero">
    <h2 class="section__title">Projects</h2>
    <p class="section__body">Things I've built and contributed to. Most are on GitHub.</p>
  </div>

  <div id="projects-grid" class="work-grid projects-grid">
    <div class="work-card skeleton" aria-hidden="true">
      <div class="skeleton__line skeleton__line--short"></div>
      <div class="skeleton__line"></div>
      <div class="skeleton__line skeleton__line--medium"></div>
    </div>
    <div class="work-card skeleton" aria-hidden="true">
      <div class="skeleton__line skeleton__line--short"></div>
      <div class="skeleton__line"></div>
      <div class="skeleton__line skeleton__line--medium"></div>
    </div>
    <div class="work-card skeleton" aria-hidden="true">
      <div class="skeleton__line skeleton__line--short"></div>
      <div class="skeleton__line"></div>
      <div class="skeleton__line skeleton__line--medium"></div>
    </div>
  </div>

  <div id="projects-error" class="projects-error" style="display:none;">
    <p>Could not load projects. <a href="https://github.com/debabratamishra">View on GitHub →</a></p>
  </div>

  <div class="projects-cta">
    <a href="https://github.com/debabratamishra" class="hero__cta-btn">View All on GitHub →</a>
  </div>
</section>

<script>
(function() {
  var grid = document.getElementById('projects-grid');
  var error = document.getElementById('projects-error');
  if (!grid) return;

  fetch('https://api.github.com/users/debabratamishra/repos?sort=updated&per_page=12')
    .then(function(r) { return r.json(); })
    .then(function(repos) {
      grid.querySelectorAll('.skeleton').forEach(function(s) { s.remove(); });
      repos.forEach(function(repo, i) {
        var card = document.createElement('a');
        card.href = repo.html_url;
        card.className = 'work-card project-card';
        card.target = '_blank';
        card.rel = 'noopener noreferrer';
        var langColor = getLangColor(repo.language);
        var metaHtml = '';
        if (repo.language) {
          metaHtml += '<span class="project-card__lang" style="--lang-color:' + langColor + '">' + repo.language + '</span>';
        }
        if (repo.stargazers_count > 0) {
          metaHtml += '<span class="project-card__stars">' + repo.stargazers_count + ' ★</span>';
        }
        card.innerHTML =
          '<span class="work-card__num">' + String(i + 1).padStart(2, '0') + '</span>' +
          '<h3 class="work-card__title">' + repo.name + '</h3>' +
          '<p class="work-card__desc">' + (repo.description || 'No description provided.') + '</p>' +
          '<div class="project-card__meta">' + metaHtml + '</div>';
        grid.appendChild(card);
      });
    })
    .catch(function() {
      grid.querySelectorAll('.skeleton').forEach(function(s) { s.remove(); });
      error.style.display = 'block';
    });

  function getLangColor(lang) {
    var colors = {
      'JavaScript': '#f1e05a', 'Python': '#3572A5', 'Rust': '#dea584',
      'TypeScript': '#3178c6', 'Ruby': '#701516', 'Go': '#00ADD8',
      'Java': '#b07219', 'C': '#555555', 'C++': '#f34b7d',
      'Jupyter Notebook': '#DA5B0B', 'Shell': '#89e051', 'HTML': '#e34c26',
      'CSS': '#563d7c', 'Makefile': '#427819'
    };
    return colors[lang] || '#888888';
  }
})();
</script>
