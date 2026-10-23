---
layout: book-review
title: Designing Distributed Systems
author: Brendan Burns
cover: assets/img/book_covers/distributed.jpg
# olid: OL43499941M # use Open Library ID to fetch cover (if no `cover` is provided)
# isbn: 7539967447 # use ISBN to fetch cover (if no `olid` is provided, dashes are optional)
categories: distributed systems
tags: top-100
# buy_link: https://www.amazon.com/Godfather-Deluxe-Mario-Puzo/dp/0593542592
# started: 2024-08-23
# finished: 2024-09-07
# released: 1969
stars: 5
# goodreads_review: 6318556633
status: reading
---

Designing Distributed Systems: Patterns and Paradigms for Scalable, Reliable Services is by Brendan Burns, a co-founder of the Kubernetes project and a Corporate Vice President at Microsoft Azure. Just as the "Gang of Four" design patterns gave object-oriented programmers a shared vocabulary, this book collects reusable patterns for building distributed systems out of containers. The **second edition** (O'Reilly) updates the original and adds new chapters on **AI inference**, **AI training**, and making systems robust in the real world.

### What it covers

- **Single-node patterns:** how containers on one machine work together: the *sidecar* (adding features such as TLS or logging to an existing app), the *ambassador* (proxying a service's communication with the outside world) and the *adapter* (standardizing monitoring and logging output).
- **Serving patterns:** replicated load-balanced services, sharded services, scatter/gather, functions-as-a-service and event-driven processing, and ownership election (leader election and distributed locks).
- **Batch patterns:** work-queue systems, event-driven batch pipelines, and coordinated batch processing such as join, reduce and MapReduce-style workflows.
- **Second-edition material:** patterns for serving and training AI models, and the practical concerns of operating systems in production, such as monitoring, failure handling and reliability.

### Why it's worth reading

Each pattern comes with a concrete example, usually deployed on Kubernetes, so the ideas move quickly from diagram to running system. It's short and practical, and especially relevant to modern ML infrastructure: model servers, request routers, sharded indexes and batch training pipelines are all built from these same patterns.
