---
layout: book-review
title: Python Concurrency with asyncio
author: Matthew Fowler
cover: assets/img/book_covers/python.png
# olid: OL43499941M # use Open Library ID to fetch cover (if no `cover` is provided)
# isbn: 7539967447 # use ISBN to fetch cover (if no `olid` is provided, dashes are optional)
categories: python asyncio
tags: top-100
# buy_link: https://www.amazon.com/Godfather-Deluxe-Mario-Puzo/dp/0593542592
# started: 2024-08-23
# finished: 2024-09-07
# released: 1969
stars: 5
# goodreads_review: 6318556633
status: paused
---

Python Concurrency with asyncio, by Matthew Fowler (Manning, 2022, about 376 pages), is a practical guide to writing concurrent Python with the standard library's `asyncio` package. Fowler is a software engineer with more than 15 years' experience in roles from architect to engineering director. The book explains how single-threaded concurrency works and when to use it instead of threads or multiple processes, and how to combine all three.

### What it covers

1. **Getting to know asyncio:** concurrency vs. parallelism, the GIL, I/O-bound vs. CPU-bound work, and how an event loop works.
2. **asyncio basics:** coroutines, `async`/`await`, tasks, timeouts and cancellation, and futures.
3. **A first application:** a non-blocking socket server built from scratch.
4. **Concurrent web requests** with `aiohttp`, `gather`, `as_completed` and `wait`.
5. **Non-blocking database drivers** with `asyncpg`, connection pools and transactions.
6. **CPU-bound work** with `multiprocessing` and process pools, including a MapReduce example.
7. **Blocking work with threads:** running legacy blocking libraries without stalling the event loop.
8. **Streams:** transports, protocols and building command-line tools.
9. **Web applications:** async web frameworks, ASGI and WebSockets.
10. **Microservices:** the backend-for-frontend pattern, retries and circuit breakers.
11. **Synchronization:** locks, semaphores, events and conditions.
12. **Asynchronous queues,** including priority and LIFO queues.
13. **Managing subprocesses** asynchronously.
14. **Advanced asyncio:** custom event loops, context variables, and how `await` really works.

### Why it's worth reading

Async Python now powers much of the modern backend, from FastAPI services to LLM agent frameworks that make many concurrent model and tool calls. The book goes beyond syntax to explain the event loop's internals, so you can reason about performance and avoid common traps such as blocking the loop.
