# LLM in C++

A lightweight, educational **Large Language Model (LLM)** implementation in **C++**, demonstrating the core building blocks of modern Transformers.

---

## Project Overview

This project implements the foundational components of a Transformer-based LLM, including:

* **Tokenizer:** Converts text into integer token IDs and decodes back.
* **Embeddings:** Maps tokens to dense vectors.
* **Positional Encoding:** Adds sequence information to embeddings.
* **Scaled Dot-Product Attention:** Core attention mechanism for learning dependencies.
* **Multi-Head Attention (MHA):** Parallel attention heads for richer token relationships.
* **Feed-Forward Network (FFN):** Simple two-layer MLP applied per token.
* **Layer Normalization & Residual Connections:** Stabilizes training and helps gradient flow.
* **Mini Transformer Block:** Combines MHA + FFN + LayerNorm + Residuals.
* **Multiple stacked transformer encoder:** Combines multiple tranformer block.
---

## Project Motivation

This project is meant for **learning and experimentation**, allowing you to:

* Understand how LLMs work from scratch.
* Experiment with Transformer math and intuition.
* Explore training and inference ideas in pure C++.

---

## Usage

1. Clone the repository:

```bash
git clone git@github.com:AmanS3109/LLM_in_Cpp.git
```

2. Compile and run the code:

```bash
g++ main.cpp -o llm -std=c++17
./llm
```

3. Observe the outputs for:

* Tokenization
* Embedding vectors
* Positional encodings
* Attention outputs
* Multi-Head Attention outputs
* Transformer block outputs

---

## Folder Structure

```
LLM_in_Cpp/
├── main.cpp            # Core code for the LLM
├── README.md           # Project documentation
├── include
├── src
```

---

## Future Work / To-Do

* Implement **training on small text corpus**.
* Save/load model weights for inference.
* Add **GPU acceleration** using CUDA.
* Support **larger vocabulary and batch processing**.

---

## References

* Vaswani et al., “Attention Is All You Need” (2017)
* Transformer and LLM tutorials in Python/PyTorch by andrej karapathy, george hotz and umar jamil
* C++ linear algebra and matrix operations references
