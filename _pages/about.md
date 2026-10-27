---
layout: about
title: about
permalink: /
subtitle: Applied AI/ML Engineer · Bengaluru

social: true # includes social icons at the bottom of the page

latest_posts:
  enabled: false
---

<section class="home-hero">
  <div class="home-hero-text">
    <p class="home-intro">
      I'm an Applied AI/ML Engineer. Professionally, most of my work has been on <strong>NLP and LLM systems</strong>:
      agent workflows, MCP servers and clients, retrieval over enterprise data, and fine-tuning and serving the models behind them.
    </p>
    <p class="home-intro">
      My interests go well beyond what I've worked on so far. I spend a lot of my own time learning other parts of AI,
      and I write up what I study in <a href="{{ '/blog/' | relative_url }}">my posts</a>.
    </p>

    <dl class="home-focus">
      <dt>At work</dt>
      <dd>
        <span class="home-chip">NLP</span>
        <span class="home-chip">LLM systems</span>
        <span class="home-chip">Agents &amp; MCP</span>
        <span class="home-chip">Retrieval &amp; RAG</span>
      </dd>
      <dt>Learning</dt>
      <dd>
        <span class="home-chip home-chip-accent">Computer vision</span>
        <span class="home-chip home-chip-accent">Inference engineering</span>
        <span class="home-chip home-chip-accent">Reinforcement learning</span>
        <span class="home-chip home-chip-accent">Embodied AI</span>
        <span class="home-chip home-chip-more">and more</span>
      </dd>
    </dl>

    <div class="home-cta">
      <a class="home-btn home-btn-primary" href="{{ '/cv/' | relative_url }}"><i class="fa-solid fa-file-lines"></i> Resume</a>
      <a class="home-btn" href="{{ '/blog/' | relative_url }}"><i class="fa-solid fa-pen-nib"></i> Posts</a>
      <a class="home-btn" href="mailto:{{ site.data.socials.email }}"><i class="fa-solid fa-envelope"></i> Get in touch</a>
    </div>
  </div>
  <div class="home-hero-photo">
    <img src="{{ '/assets/img/prof_pic.jpg' | relative_url }}" alt="Narayan Jangid" loading="eager">
    <p class="home-location"><i class="fa-solid fa-location-dot"></i> Bengaluru, Karnataka</p>
  </div>
</section>

<section class="home-section">
  <h2 class="home-h2">experience <span class="home-total">{% include duration.liquid start="2023-06" %} in total</span></h2>

  <ol class="xp">
    <li class="xp-item xp-current">
      <div class="xp-when">Apr 2026 – Present<span>{% include duration.liquid start="2026-04" %}</span></div>
      <div class="xp-body">
        <h3 class="xp-company">Recruiterflow</h3>
        <div class="xp-role">
          <h4>Data Engineer (AI Infra)</h4>
          <ul>
            <li>Architected the Natural Language Search data pipeline, processing 200K+ candidates and contacts and 10K+ jobs and deals a day to power LLM-driven search.</li>
            <li>Migrated the company-enrichment pipeline off a legacy system, adding caching and request optimizations that cut API costs by over 2.5×.</li>
          </ul>
        </div>
      </div>
    </li>
    <li class="xp-item">
      <div class="xp-when">Mar 2025 – Apr 2026<span>{% include duration.liquid months=13 %}</span></div>
      <div class="xp-body">
        <h3 class="xp-company">Kloudspot</h3>
        <div class="xp-role">
          <h4>Machine Learning Engineer</h4>
          <ul>
            <li>Led the end-to-end design of LISA AI, an enterprise AI assistant, and managed and mentored a 5-member AI team.</li>
            <li>Benchmarked inference strategies and tuned vLLM/SGLang with EAGLE-3 speculative decoding for up to 2× throughput and 2.7–3.5× lower latency.</li>
          </ul>
        </div>
      </div>
    </li>
    <li class="xp-item">
      <div class="xp-when">Jun 2023 – Feb 2025<span>{% include duration.liquid months=21 %}</span></div>
      <div class="xp-body">
        <h3 class="xp-company">Celebal Technologies</h3>
        <div class="xp-role">
          <h4>Associate Data Scientist <span class="xp-role-date">Nov 2023 – Feb 2025</span></h4>
          <ul>
            <li>Led Azure OpenAI and Document Intelligence solutions to extract, embed and index enterprise documents, improving BLEU by 15%.</li>
            <li>Designed a hybrid edge–cloud VLM inference system for real-time video analytics, reaching 77.9% mAP.</li>
          </ul>
        </div>
        <div class="xp-role">
          <h4>Data Science Intern <span class="xp-role-date">Jun 2023 – Oct 2023</span></h4>
          <ul>
            <li>Fine-tuned GPT-3.5-turbo and Mistral 7B for text-to-query generation (SQL, NoSQL), with DataCompy scores above 95%.</li>
            <li>Improved a RAG pipeline over tabular data, raising search-result accuracy from 60% to 85%.</li>
          </ul>
        </div>
      </div>
    </li>
  </ol>
  <p class="home-more"><a href="{{ '/cv/' | relative_url }}">View full resume →</a></p>
</section>

<section class="home-section">
  <h2 class="home-h2">education</h2>
  <div class="home-edu-card">
    <i class="fa-solid fa-graduation-cap home-edu-icon"></i>
    <div class="home-edu-text">
      <h3>IISER Kolkata</h3>
      <p>BS-MS Dual Degree in Geological Sciences, Minor in Physics</p>
      <a href="{{ '/research/' | relative_url }}">Master's thesis →</a>
    </div>
    <span class="home-edu-date">2017 – 2022</span>
  </div>
</section>
