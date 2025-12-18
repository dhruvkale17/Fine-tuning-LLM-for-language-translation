## Fine-Tuning Qwen2.5 for German → French Translation using QLoRA

This repository demonstrates an end-to-end **LLM fine-tuning pipeline for German-to-French machine translation** using **parameter-efficient fine-tuning (QLoRA)** under limited GPU resources.

The project covers dataset selection, prompt engineering, model fine-tuning, synthetic data augmentation, and evaluation using both lexical and semantic translation metrics.

---

### Project Overview

* **Base Model:** Qwen2.5-1.5B-Instruct
* **Task:** German → French language translation
* **Fine-Tuning Method:** QLoRA (4-bit quantized LoRA adapters)
* **Platform:** Google Colab (15 GB GPU constraint)
* **Dataset:** Opus Books (Hugging Face)
* **Evaluation Metrics:** BLEU & COMET

The goal of this project is to **improve translation quality while staying within strict compute and memory limits**, simulating a real-world production or research constraint.

---

### Dataset

* **Primary Dataset:** Opus Books (German–French parallel corpus)
* Professionally translated, copyright-free book text
* Rich vocabulary and diverse sentence structures
* 1,000 samples selected and split into **80% train / 20% test**

#### Synthetic Data Augmentation

To study the impact of data augmentation:

* A **synthetic dataset (1,600 samples)** was generated using **Llama3-70B**
* Few-shot, instruction-based prompting ensured:

  * Complex, non-repetitive sentences
  * Multiple domains (literature, science, technology, daily life)
  * Strict parallel sentence formatting
* Synthetic data was evaluated both independently and combined with the original dataset

---

### Model & Fine-Tuning Strategy

* **Qwen2.5-1.5B-Instruct** was selected for its:

  * Multilingual support (German & French)
  * Instruction-following capability
  * Strong generative performance

* **QLoRA** was used to:

  * Enable fine-tuning in limited GPU memory
  * Preserve pre-trained knowledge
  * Reduce training time and computational cost
  * Update only low-rank adapter layers instead of full model weights

---

### Prompt Engineering

Two instruction-based prompts were designed:

1. **Translation Prompt** (for fine-tuning & inference)
   Ensures consistent input/output structure for supervised learning.

2. **Synthetic Data Generation Prompt**
   Uses strict formatting rules, few-shot examples, and linguistic constraints to maximize data quality.

---

### Experiments & Evaluation

Four models were trained and evaluated:

| Model   | Description                          | BLEU       | COMET      |
| ------- | ------------------------------------ | ---------- | ---------- |
| Model A | Base model (no fine-tuning)          | 0.0151     | 0.398      |
| Model B | Fine-tuned on original dataset       | 0.0254     | **0.4276** |
| Model C | Fine-tuned on synthetic dataset only | 0.0230     | 0.3718     |
| Model D | Fine-tuned on combined dataset       | **0.0279** | 0.3976     |

* **BLEU** captures n-gram overlap
* **COMET** captures semantic correctness

Results show that:

* Fine-tuning improves translation quality
* Synthetic data alone may introduce noise
* Combining real + synthetic data improves BLEU but not necessarily semantic alignment

---

### Implementation Highlights

* Fully modular and well-documented Colab notebook
* Clear experiment flow with reproducible steps
* Automated evaluation and visualization
* Matplotlib & Seaborn used for performance comparison
* All dependencies and configurations included

---

### Key Takeaways

* QLoRA enables effective LLM fine-tuning on consumer-grade GPUs
* Prompt design significantly impacts both learning and data quality
* Synthetic data must be carefully curated to avoid degrading semantic performance
* Using complementary evaluation metrics provides deeper insight into translation quality

---

This project serves as a **practical reference for LLM fine-tuning, multilingual NLP, and parameter-efficient training strategies** under real-world constraints.
