---
layout: post
title: Understanding Transformers — The Architecture Behind Modern LLMs
date: 2025-08-31 14:00:00
description: A deep dive into the Transformer architecture — self-attention, multi-head attention, position encodings, and scaling laws.
tags: transformers deep-learning llms nlp
categories: ai-ml
featured: true
thumbnail: assets/img/transformer-diagram.png
---

The **Transformer architecture** is the backbone of modern **Large Language Models (LLMs)** like GPT, BERT, and T5.  
Introduced in the landmark paper [“Attention Is All You Need” (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), it replaced RNNs and CNNs in sequence modeling with a highly parallelizable **self-attention mechanism**.

This post breaks down the architecture and all its nitty-gritty components.

---

## 🚀 Motivation

Before Transformers, we had **RNNs** and **LSTMs** for sequence modeling.  
But they suffered from:

- Sequential processing (slow, hard to parallelize)
- Vanishing/exploding gradients
- Difficulty handling **long-range dependencies**

Transformers solved these with **attention**, allowing models to *look at all tokens simultaneously*.

---

## 🧩 Transformer Components

At a high level, the Transformer consists of **an Encoder** and **a Decoder**.

#### Encoder
- Takes the input sequence (e.g., words, tokens).
- Builds contextualized embeddings.

#### Decoder
- Generates the output sequence step by step.
- Used in tasks like **machine translation** or **text generation**.

---

### 🔑 Input Embeddings & Positional Encoding

Since Transformers don’t have recurrence like RNNs, they need a way to encode *word order*.  
They use **positional encodings** (sinusoidal or learned vectors).

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2*(i//2))/d_model)

    angle_rads = pos * angle_rates
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])  # even indices
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])  # odd indices
    return pe
```

---

### 🌀 Self-Attention

The **core innovation** of Transformers.  
Each word *attends* to every other word to capture contextual meaning.

The formula:

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:
- **Q** = Queries  
- **K** = Keys  
- **V** = Values  

#### Example:
If the sentence is *“The cat sat on the mat”*,  
- The word **“cat”** can attend strongly to **“sat”**, **“mat”**, etc.  

---

### 🎭 Multi-Head Attention

Instead of one attention mechanism, we use multiple in parallel (heads).  
This allows the model to learn **different types of relationships**.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.qkv_proj = torch.nn.Linear(d_model, 3*d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)
        self.num_heads = num_heads

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, self.num_heads, 3*self.d_head)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, T, C)
        return self.o_proj(out)
```

---

### 🔄 Feed Forward Networks (FFN)

Each layer also has a simple MLP:  
Two linear layers with a **ReLU/GELU activation** in between.

---

### ⚖️ Layer Normalization & Residuals

- Residual connections help stabilize training.  
- Layer normalization ensures smoother gradient flow.

---

## 📚 Scaling Laws & Training

Recent research shows that Transformer performance improves predictably with:

- **Model size** (parameters)
- **Dataset size**
- **Compute**

This led to **scaling laws** powering GPT-3, PaLM, and other frontier models.

---

## ✅ Checklist for Understanding

- [x] Why RNNs struggled  
- [x] Transformer Encoder-Decoder overview  
- [x] Positional Encoding  
- [x] Self-Attention  
- [x] Multi-Head Attention  
- [x] Feed Forward Networks  
- [x] Residuals + LayerNorm  

---

> *"Attention is not just a mechanism, it’s the foundation of intelligence in sequence models."*  
> — Inspired by Vaswani et al., 2017

---

## 🖼️ Diagram

You can include the classic architecture diagram here:

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

---

## ⚡ Final Thoughts

The Transformer revolutionized deep learning.  
From **translation** to **LLMs** like GPT-4, Claude, and LLaMA,  
its *self-attention-first* design is the core reason behind today’s AI boom.

---