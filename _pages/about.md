---
layout: about
title: about
permalink: /
subtitle: Applied AI/ML Engineer · Agentic systems, retrieval, inference engineering & reinforcement learning · Bengaluru

social: true # includes social icons at the bottom of the page

latest_posts:
  enabled: false
---

<section class="home-hero">
  <div class="home-hero-text">
    <p class="home-eyebrow"><span class="home-dot"></span> Data Engineer (AI Infra) at <a href="https://recruiterflow.com" target="_blank" rel="noopener noreferrer">Recruiterflow</a></p>
    <h2 class="home-headline">I take ML and GenAI systems from prototype to production.</h2>
    <p>
      I'm an Applied AI/ML Engineer with <strong>3+ years</strong> of shipping production ML, including
      <strong>1.5+ years building agentic systems</strong>: multi-step agent workflows, MCP servers and clients,
      and retrieval pipelines over enterprise data. I pair classical ML (time-series forecasting, gradient-boosted
      trees) with LLM fine-tuning, inference optimization, and metric-driven evaluation, and I've led a 5-member AI team.
    </p>
    <div class="home-cta">
      <a class="home-btn home-btn-primary" href="{{ '/cv/' | relative_url }}"><i class="fa-solid fa-file-lines"></i> Resume</a>
      <a class="home-btn" href="mailto:{{ site.data.socials.email }}"><i class="fa-solid fa-envelope"></i> Get in touch</a>
    </div>
  </div>
  <div class="home-hero-photo">
    <img src="{{ '/assets/img/prof_pic.jpg' | relative_url }}" alt="Narayan Jangid" loading="eager">
    <p class="home-location"><i class="fa-solid fa-location-dot"></i> Bengaluru, Karnataka</p>
  </div>
</section>

<h2 class="home-h2">what I work on</h2>

- **Agentic systems & MCP:** multi-step agent workflows in LangGraph, graph-based reasoning, and MCP servers and clients that expose real-time business data as agent tools.
- **Retrieval & search:** natural-language search pipelines, OpenSearch indexing and retrieval at multi-tenant scale, and RAG over enterprise data.
- **Inference engineering & fine-tuning:** vLLM/SGLang tuning, speculative decoding, latency–throughput benchmarking, and instruction fine-tuning (SFT, QLoRA).
- **Classical ML & forecasting:** time-series forecasting (ARIMA/SARIMA), gradient-boosted trees, document intelligence, OCR, and vision-language models.

<section class="home-section">
  <h2 class="home-h2">experience <span class="home-total">{% include duration.liquid start="2023-06" %} in total</span></h2>
  <ol class="home-timeline">
    <li>
      <div class="home-tl-head"><strong>Data Engineer (AI Infra)</strong> · Recruiterflow <span class="home-tl-date">Apr 2026 – Present · {% include duration.liquid start="2026-04" %}</span></div>
      <p>Built the Recruiterflow MCP server for Claude, the Natural Language Search data pipeline, and a rebuilt OpenSearch Omni-Search with shared indexes.</p>
    </li>
    <li>
      <div class="home-tl-head"><strong>Machine Learning Engineer</strong> · Kloudspot <span class="home-tl-date">Mar 2025 – Apr 2026 · {% include duration.liquid months=13 %}</span></div>
      <p>Led the design of LISA AI, an enterprise AI assistant, and a 5-member AI team. Built agentic workflows and an MCP client, and tuned LLM serving for up to 2× throughput.</p>
    </li>
    <li>
      <div class="home-tl-head"><strong>Associate Data Scientist</strong> · Celebal Technologies <span class="home-tl-date">Nov 2023 – Feb 2025 · {% include duration.liquid months=16 %}</span></div>
      <p>Built energy demand forecasting, Azure OpenAI document intelligence, an on-prem OCR pipeline, and hybrid edge–cloud VLM inference (77.9% mAP).</p>
    </li>
    <li>
      <div class="home-tl-head"><strong>Data Science Intern</strong> · Celebal Technologies <span class="home-tl-date">Jun 2023 – Oct 2023 · {% include duration.liquid months=5 %}</span></div>
      <p>Fine-tuned GPT-3.5 and Mistral 7B for text-to-SQL/NoSQL (95%+ DataCompy). Raised tabular RAG accuracy from 60% to 85%.</p>
    </li>
  </ol>
  <p class="home-more"><a href="{{ '/cv/' | relative_url }}">View full resume →</a></p>
</section>

<section class="home-section">
  <h2 class="home-h2">toolbox</h2>
  <div class="home-chips">
    {% assign tools = "Python,PyTorch,LangGraph,LangChain,LlamaIndex,MCP,RAG,vLLM,SGLang,PEFT/QLoRA,Transformers,XGBoost,LightGBM,OpenSearch,Elasticsearch,Redis,Celery,FastAPI,Flask,PySpark,Databricks,Docker,AWS,Azure,PostgreSQL,MongoDB" | split: "," %}
    {% for t in tools %}<span class="home-chip">{{ t }}</span>{% endfor %}
  </div>
</section>

<section class="home-section">
  <h2 class="home-h2">education &amp; certifications</h2>
  <div class="home-edu">
    <div>
      <h4>IISER Kolkata</h4>
      <p>BS-MS Dual Degree in Geological Sciences, Minor in Physics · 2017 – 2022</p>
    </div>
    <ul class="home-certs">
      <li><i class="fa-brands fa-microsoft"></i> Azure Data Scientist Associate</li>
      <li><i class="fa-brands fa-microsoft"></i> Azure AI Engineer Associate</li>
      <li><i class="fa-solid fa-certificate"></i> Databricks Generative AI Engineer Associate</li>
    </ul>
  </div>
</section>
