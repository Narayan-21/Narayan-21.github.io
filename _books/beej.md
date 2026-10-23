---
layout: book-review
title: Beej's Guide to C Programming
author: Brian “Beej Jorgensen” Hall
cover: assets/img/book_covers/beej.jpg
# olid: OL43499941M # use Open Library ID to fetch cover (if no `cover` is provided)
# isbn: 7539967447 # use ISBN to fetch cover (if no `olid` is provided, dashes are optional)
categories: c
tags: top-100
# buy_link: https://www.amazon.com/Godfather-Deluxe-Mario-Puzo/dp/0593542592
# started: 2024-08-23
# finished: 2024-09-07
# released: 1969
stars: 5
# goodreads_review: 6318556633
status: finished
---

Beej's Guide to C Programming is a free, continuously updated guide to C by Brian "Beej Jorgensen" Hall, the author of the well-known *Beej's Guide to Network Programming*. It's written for people who already know another language and want to learn C properly, in the same informal, joke-filled style as his other guides. The latest release (v0.10.x, 2026) runs to more than forty chapters.

### What it covers

The guide comes in two volumes:

- **Tutorial:** starts with "Hello, World!" and the build process, then covers variables, operators and control flow, functions, pointers and pointer arithmetic, arrays, strings, structs, file I/O, `typedef`, manual memory allocation, scope and storage classes, and the preprocessor.
- **Advanced topics:** later chapters cover the parts of C most tutorials skip: variadic functions, locales and Unicode/wide characters, `setjmp`/`longjmp`, signal handling, variable-length arrays, `goto`, type qualifiers, compound literals and generic selections, multithreading with `<threads.h>`, and atomics and the C memory model.
- **Library reference:** a separate volume documents the C standard library, header by header, with examples.

### Why it's worth reading

C is the language under almost everything else: operating systems, Python's interpreter, CUDA's host code, database engines. This guide explains *why* C behaves the way it does (memory layout, undefined behavior, what a pointer really is) rather than just listing syntax, and it targets modern C (C11/C23) instead of older textbook C. It's freely available in HTML, PDF and print at [beej.us/guide/bgc](https://beej.us/guide/bgc/).
