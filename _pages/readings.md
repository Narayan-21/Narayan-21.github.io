---
layout: default
permalink: /readings/
title: Readings
nav: true
nav_order: 2
pagination:
  enabled: true
  collection: readings
  permalink: /readings/page/:num/
  per_page: 15
  sort_field: date
  sort_reverse: true
---

<div class="readings">
  <div class="header-bar">
    <h1>Readings</h1>
    <h2>Interesting articles, papers, and resources</h2>
  </div>

  <ul class="reading-list">
    {% if page.pagination.enabled %}
      {% assign readinglist = paginator.posts %}
    {% else %}
      {% assign readinglist = site.readings %}
    {% endif %}

    {% for item in readinglist %}
      <li class="reading-item">
        <span class="reading-date">{{ item.date | date: '%b %d, %Y' }}</span>
        <a href="{% if item.redirect %}{{ item.redirect }}{% else %}{{ item.url | relative_url }}{% endif %}" 
           {% if item.redirect contains '://' %}target="_blank"{% endif %}>
          {{ item.title }}
        </a>
      </li>
    {% endfor %}
  </ul>

  {% if page.pagination.enabled %}
    {% include pagination.liquid %}
  {% endif %}
</div>

<style>
.reading-list {
  list-style: none;
  padding-left: 0;
}

.reading-item {
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #eee;
}

.reading-date {
  display: inline-block;
  min-width: 100px;
  color: #666;
  font-size: 0.9em;
}

.reading-item a {
  text-decoration: none;
  font-weight: 500;
}

.reading-item a:hover {
  text-decoration: underline;
}
</style>