---
layout: book-review
title: CUDA by Example
author: Jason Sanders and Edward Kandrot
cover: assets/img/book_covers/cuda.jpg
# olid: OL43499941M # use Open Library ID to fetch cover (if no `cover` is provided)
# isbn: 7539967447 # use ISBN to fetch cover (if no `olid` is provided, dashes are optional)
categories: parallel_programming
tags: top-100
# buy_link: https://www.amazon.com/Godfather-Deluxe-Mario-Puzo/dp/0593542592
# started: 2024-08-23
# finished: 2024-09-07
# released: 1969
stars: 5
# goodreads_review: 6318556633
status: reading
---

CUDA by Example: An Introduction to General-Purpose GPU Programming, by Jason Sanders and Edward Kandrot (both senior members of NVIDIA's CUDA team), was published by Addison-Wesley in 2010. It's one of the gentlest introductions to GPU programming: rather than starting with hardware theory, it teaches CUDA C through small, complete programs you can run and modify.

### What it covers

- **Foundations:** why GPUs became general-purpose processors, setting up a CUDA environment, and the basics of CUDA C: kernels, `__global__` functions, device memory, and launching work from the host.
- **Parallelism:** threads, blocks and grids; how to map a problem onto thousands of threads; and examples such as vector addition, dot products and Julia-set fractals.
- **Thread cooperation:** shared memory, `__syncthreads()`, and how threads in a block share work.
- **The GPU memory system:** constant memory, texture memory, and using CUDA events to measure performance.
- **Advanced topics:** graphics interoperability with OpenGL, atomic operations (including a GPU hash table and histogram), streams for overlapping computation with data transfer, zero-copy host memory, and using multiple GPUs.

### Why it's worth reading

Some APIs and hardware details are dated (the book predates Tensor Cores and modern architectures), but the core ideas haven't changed: grids and blocks, the memory hierarchy, synchronization, atomics and streams. It's a quick, hands-on first step before moving to a deeper text like *Programming Massively Parallel Processors*, and a good base for understanding how deep-learning kernels and inference engines use the GPU.
