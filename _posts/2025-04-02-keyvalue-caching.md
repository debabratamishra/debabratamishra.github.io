---
title: 'Key-Value caching'
layout: single
author_profile: true
date: 2025-05-02
permalink: /posts/2025/05/keyvalue-caching/
tags:
  - Large Language Models
  - Artificial Intelligence
comments: true
---

## What is Key-Value caching?
Key-value caching, as an optimization technique, focuses on improving the efficiency of the inference process in Large Language Models(LLMs) by reusing previously computed states. In simple terms, it's a way for the model to "remember" previous calculations to avoid re-computing them for every new word it generates.

Imagine you're having a conversation. You don't re-process the entire conversation from the beginning every time someone says something new. Instead, you maintain the context and build upon it. KV caching works on a similar principle for LLMs.

To understand this better, we will briefly touch upon the Transformer architecture proposed in the gamous 2017 paper "Attention is All you need", the foundation of most modern LLMs.

### Transformer architecture overview

From a high-level perspective, most transformers consist of a few basic building blocks:

- A tokenizer that splits the input text into subparts, such as words or sub-words.
- An embedding layer that transforms the resulting tokens (and their relative positions within the texts) into vectors.
- A couple of basic neural network layers, including dropout, layer normalization, and regular feed-forward linear layers.

The most innovative of these building blocks is the self-attention mechanism. This mechanism allows the model to weigh the importance of different words in the input sequence when producing the next word. This is where the concepts of Keys and Values originate, and it's the core of what KV Caching optimizes.

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/images/Transformers.png" alt="Transformer Architecture" style="max-width: 40%; height: auto;">
  <p style="font-style: italic; font-size: 0.9em; text-align: center;">Figure 1: Model Architecture of Transformer<sup><a href="https://arxiv.org/abs/1706.03762v7">[1]</a></sup></p>
</div>

### A Closer Look at Self-Attention

Let's zoom in and understand how self-attention works. For every input token that has been converted into an embedding vector, the model generates three new, distinct vectors:

- Query $(Q)$: Think of this as the current token's "search query." It's looking for relevant information from other tokens in the sequence to better understand its own context.
- Key $(K)$: This is like a "label" or an "index" for a token. It's what the Query vector from other tokens will match against to find relevant information.
- Value $(V)$: This vector contains the actual substance or meaning of the token. Once a Query finds a relevant Key, the associated Value is what provides the useful information.

These $Q$, $K$, and $V$ vectors are created by multiplying the token's embedding vector (say $x$) by three separate weight matrices($W^Q, W^K, W^V$). These matrices are learned during the model's training and are essential for its performance. Basically for an input vector $x$ the process would be :
- $Query = x.W^Q$
- $Key = x.W^K$
- $Value = x.W^V$

The model then uses the Query vector of the current token to score itself against the Key vectors of all tokens in the sequence (including itself). These scores determine how much "attention" to pay to each token's Value, and this is what makes the model context-aware.

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/images/multihead_attention.png" alt="Multi-Head Attention" style="max-width: 30%; height: auto;">
  <p style="font-style: italic; font-size: 0.9em; text-align: center;">Figure 2: The Multi-Head Attention block in a Transformer<sup><a href="https://arxiv.org/abs/1706.03762v7">[1]</a></sup></p>
</div>

### How Does Key-Value Caching Work?

LLMs generate text in an autoregressive manner, meaning they generate one word at a time, and each new word depends on the previously generated words. Without KV caching, every time the model generates a new word, it would have to recalculate the Key and Value vectors for all the preceding words in the sequence. This is incredibly inefficient and computationally expensive, especially for long sequences of text.

This is where KV caching comes to the rescue.

Instead of discarding the Key and Value vectors after they are calculated, the model stores them in a cache. When generating the next word, the model only needs to calculate the $Q, K$ and $V$ vectors for the newest word and can then retrieve the $K$ and $V$ vectors of all the previous words directly from the cache.

Let's illustrate with an example:

Suppose we want the model to complete the sentence: "The quick brown fox..."

1. First Word ("The"): The model processes "The" and calculates its K and V vectors. These are then stored in the KV cache.
2. Second Word ("quick"): The model processes "quick." It calculates the Q vector for "quick" and retrieves the K and V vectors for "The" from the cache. It then calculates the K and V for "quick" and adds them to the cache.
3. Third Word ("brown"): The model processes "brown." It calculates the Q vector for "brown" and retrieves the K and V vectors for "The" and "quick" from the cache. The new K and V for "brown" are also cached.
4. Fourth Word ("fox"): The process repeats. The model only needs to compute the Q, K, and V for "fox" and can reuse the cached K and V vectors for "The quick brown."

This caching mechanism dramatically speeds up the generation of subsequent tokens.


### Why is Key-Value Caching Needed?
The primary need for KV caching boils down to two main factors: speed and efficiency.

- Reduced Latency

By avoiding redundant calculations, KV caching significantly reduces the time it takes to generate each new token. This is why you see a much faster output after the initial prompt processing when using an LLM. The initial "thinking" time is partly the model calculating the KV cache for your input prompt.

- Maintaining Context

In multi-turn conversations or long documents, KV caching helps maintain context efficiently without recomputing everything.

- Lower Computational Cost

Re-calculating the Key and Value vectors for the entire sequence for every new token would be incredibly resource-intensive, requiring a massive amount of computational power and leading to higher operational costs.

However, there is a trade-off.

### The Memory Trade-Off

The significant advantage in speed comes at the cost of increased memory usage. The KV cache for a long sequence can become quite large, as the model needs to store the Key and Value vectors for every token in the context. This memory consumption can be a bottleneck, especially when dealing with very long text sequences or serving multiple users simultaneously.

But there are various ways to optimize the KV cache, such as:

- Quantization: Reducing the precision of the numbers stored in the cache to save space.
- Eviction Policies: Intelligently deciding which parts of the cache to discard when it gets too full.



### References
1. [Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017)](https://arxiv.org/abs/1706.03762v7)

2. [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)

3. [Key-Value Caching](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies/key-value-kv-caching)