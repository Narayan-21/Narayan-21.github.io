---
layout: default
permalink: /blog/
title: posts
nav: true
nav_order: 1
---

{% comment %}
  Posts: every post, newest first, with a topic sidebar (a post's tags) and a sort switch.
  Filtering and sorting run client-side, so the full list is rendered here without pagination;
  without JavaScript the page is still a complete, newest-first list.
{% endcomment %}

{% assign posts = site.posts | where_exp: "post", "post.hidden != true" %}

{% assign topics = site.data.post_topics %}
{% if topics == nil %}{% assign topics = "" | split: "," %}{% endif %}
{% for tag in site.tags %}
  {% unless topics contains tag[0] %}
    {% assign topics = topics | push: tag[0] %}
  {% endunless %}
{% endfor %}

<div class="post posts-page">
  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <p class="lead">{{ site.blog_description }}</p>
  </div>

  <div class="posts-layout">
    <div class="posts-main">
      <p class="posts-status" aria-live="polite"></p>

      <ul class="posts-list">
        {% for post in posts %}
          {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
          {% capture post_topics %}{% for tag in post.tags %}{{ tag | slugify }} {% endfor %}{% endcapture %}
          <li class="posts-item" data-topics="{{ post_topics | strip }}" data-date="{{ post.date | date: '%s' }}">
            <p class="posts-meta">
              <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%b %d, %Y' }}</time>
              &middot; {{ read_time }} min read
            </p>
            <h3 class="posts-title">
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            </h3>
            {% if post.description %}
              <p class="posts-description">{{ post.description }}</p>
            {% endif %}
            {% if post.tags.size > 0 %}
              <p class="posts-tags">
                {% for tag in post.tags %}
                  <a class="posts-tag" href="{{ '/blog/' | relative_url }}?topic={{ tag | slugify }}" data-topic="{{ tag | slugify }}">{{ tag }}</a>
                {% endfor %}
              </p>
            {% endif %}
          </li>
        {% endfor %}
      </ul>

      <p class="posts-empty" hidden>
        Nothing here on this topic yet. <a href="{{ '/blog/' | relative_url }}" data-topic="">See all posts</a>.
      </p>
    </div>

    <aside class="posts-sidebar" aria-label="Filter and sort posts">
      <h2 class="posts-sidebar-heading">Topics</h2>
      <div class="posts-topics" role="group" aria-label="Topics">
        <button type="button" class="posts-chip" data-topic="" aria-pressed="true">
          <span class="posts-chip-name">All</span> <span class="posts-count">{{ posts.size }}</span>
        </button>
        {% for topic in topics %}
          {% assign topic_count = site.tags[topic] | size %}
          <button
            type="button"
            class="posts-chip"
            data-topic="{{ topic | slugify }}"
            aria-pressed="false"
            {% if topic_count == 0 %}
              disabled title="Nothing here yet"
            {% endif %}
          >
            <span class="posts-chip-name">{{ topic }}</span> <span class="posts-count">{{ topic_count }}</span>
          </button>
        {% endfor %}
      </div>

      <h2 class="posts-sidebar-heading">Order</h2>
      <div class="posts-sort">
        <label for="posts-sort-select" class="sr-only">Sort</label>
        <select id="posts-sort-select" class="posts-select">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
        <a class="posts-rss" href="{{ '/feed.xml' | relative_url }}" title="RSS feed" aria-label="RSS feed"><i class="fa-solid fa-square-rss"></i></a>
      </div>
    </aside>
  </div>
</div>

<script>
  (function () {
    var list = document.querySelector(".posts-list");
    var items = Array.prototype.slice.call(list.querySelectorAll(".posts-item"));
    var chips = Array.prototype.slice.call(document.querySelectorAll(".posts-chip"));
    var select = document.getElementById("posts-sort-select");
    var status = document.querySelector(".posts-status");
    var empty = document.querySelector(".posts-empty");
    var labels = {};
    chips.forEach(function (c) {
      labels[c.dataset.topic] = c.querySelector(".posts-chip-name").textContent.trim();
    });

    var params = new URLSearchParams(window.location.search);
    var state = {
      topic: labels.hasOwnProperty(params.get("topic")) ? params.get("topic") : "",
      sort: params.get("sort") === "oldest" ? "oldest" : "newest",
    };

    function render() {
      chips.forEach(function (c) {
        c.setAttribute("aria-pressed", String(c.dataset.topic === state.topic));
      });
      select.value = state.sort;

      items.sort(function (a, b) {
        var d = Number(b.dataset.date) - Number(a.dataset.date);
        return state.sort === "oldest" ? -d : d;
      });
      var shown = 0;
      items.forEach(function (item) {
        var match = !state.topic || item.dataset.topics.split(" ").indexOf(state.topic) !== -1;
        item.hidden = !match;
        if (match) shown++;
        list.appendChild(item);
      });

      empty.hidden = shown > 0;
      var noun = shown === 1 ? "post" : "posts";
      status.textContent = state.topic ? shown + " " + noun + " on " + labels[state.topic] : "";

      var q = new URLSearchParams();
      if (state.topic) q.set("topic", state.topic);
      if (state.sort === "oldest") q.set("sort", "oldest");
      var qs = q.toString();
      history.replaceState(null, "", window.location.pathname + (qs ? "?" + qs : ""));
    }

    function pick(topic) {
      state.topic = topic;
      render();
    }

    chips.forEach(function (c) {
      c.addEventListener("click", function () {
        pick(c.dataset.topic === state.topic ? "" : c.dataset.topic);
      });
    });
    document.querySelectorAll(".posts-tag, .posts-empty a").forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        pick(a.dataset.topic);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });
    select.addEventListener("change", function () {
      state.sort = select.value;
      render();
    });

    render();
  })();
</script>
