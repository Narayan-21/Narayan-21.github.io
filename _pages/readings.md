---
layout: default
permalink: /readings/
title: Readings
nav: true
nav_order: 2
---

<div class="readings">
  <div class="header-bar">
    <h1>Readings</h1>
    <p class="lead">A collection of interesting papers and articles</p>
  </div>

  <ul class="reading-list">
    {% for item in site.readings %}
      <li class="reading-item">
        <a class="reading-title" href="{% if item.redirect %}{{ item.redirect }}{% else %}{{ item.url | relative_url }}{% endif %}" 
           {% if item.redirect contains '://' %}target="_blank"{% endif %}>
          {{ item.title }}
        </a>
        {% if item.description %}
          <p class="reading-description">{{ item.description }}</p>
        {% endif %}
      </li>
    {% endfor %}
  </ul>
</div>

<style>
.reading-list {
  list-style: none;
  padding-left: 0;
}

.reading-item {
  margin-bottom: 2rem;
}

.reading-title {
  font-size: 1.2rem;
  font-weight: 500;
  text-decoration: none;
}

.reading-description {
  opacity: 0.7;
  margin: 0.25rem 0 0 0;
  font-size: 0.9rem;
}
</style>