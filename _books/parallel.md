---
layout: book-review
title: Programming Massively Parallel Processors
author: Wen-mei W. Hwu, David B. Kirk and Izzat El Hajj
cover: assets/img/book_covers/parallel.jpg
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

Programming Massively Parallel Processors: A Hands-on Approach, by Wen-mei W. Hwu, David B. Kirk and Izzat El Hajj, is widely regarded as *the* textbook on GPU computing. Hwu is a professor emeritus at the University of Illinois and a Senior Director of Research at NVIDIA, and Kirk is a former NVIDIA Chief Scientist. The book grew out of their parallel-programming courses at Illinois. The latest **5th edition** (Morgan Kaufmann, 2026) adds more optimizations, new patterns and applications, and wider coverage of CUDA features.

### What it covers

- **Fundamentals:** heterogeneous computing, CUDA's data-parallel programming model, multidimensional grids and data, compute architecture and scheduling, warps and SIMD execution, and the memory hierarchy (global, shared and constant memory, plus caches).
- **Performance:** memory coalescing, tiling, occupancy, thread coarsening, latency hiding, and a checklist of common optimizations, with every technique motivated by how the hardware actually works.
- **Parallel patterns:** convolution, stencils, histograms (privatization and atomics), reduction, prefix sum (scan), merge, sorting, sparse matrix computation and graph traversal, each built up from a naive version to an optimized one.
- **Applications and advanced practice:** deep learning (convolutional layers as matrix multiplication), MRI reconstruction, electrostatic potential maps, iterative solvers, dynamic parallelism, and programming GPU clusters with CUDA and MPI.

### Why it's worth reading

This is the book to read after a first CUDA tutorial. It teaches you to *think* in parallel and to reason about performance from first principles, which is exactly the skill behind writing fast attention kernels, GEMMs and inference engines. Its exercises and case studies make it suitable for self-study as well as a course.
