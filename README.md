# NLU Assignment 2: Word Embeddings and Sequence Modeling
Pragati Rokade (B23CM1055)

This repository contains the implementation for CSL 7640: Natural Language Understanding, Assignment 2. The project is divided into two primary problems: learning domain-specific word embeddings and character-level sequence modeling for name generation.

---

## Problem 1: Word Embeddings from IIT Jodhpur Data

### Objective
To train and analyze Word2Vec models (CBOW and Skip-gram) on a custom corpus derived from IIT Jodhpur digital sources.

### Dataset & Preprocessing
The dataset was constructed from 7 distinct sources, including the B.Tech programs page, CSE department pages, and official Academic Regulations.
- **Total Documents**: 7
- **Total Tokens**: 33,386
- **Vocabulary Size**: 5,671

**Preprocessing Pipeline:**
1. Removal of HTML/PDF boilerplate.
2. Conversion to lowercase.
3. Removal of punctuation and non-alphabetic characters.
4. Tokenization and stop-word removal.

### Semantic Analysis (w2v_cbow_d50_w5_n10.model)
The model successfully captured domain-specific relationships within the institute's context.
- **Analogy**: BTech : Bachelors :: MTech : masters (Score: 0.6337)
- **Nearest Neighbors**: 
  - *research*: contribute, mobisys, papers, multidisciplinary
  - *examination*: makeup, comprehensive, perform, attempt

### Visualization Interpretation
- **CBOW**: Tends to smooth fine-grained details, resulting in dense, isolated clusters for specific domains like medical terms or engineering branches.
- **Skip-gram**: Preserves individual semantic nuance, showing more dispersed clusters that acknowledge the different functional roles of related terms (e.g., "hospital" vs. "doctor").

---

## Problem 2: Character-Level Name Generation

### Objective
Design and compare three recurrent neural architectures for generating realistic Indian names.

### Model Architectures
1. **Vanilla RNN**: Standard architecture using tanh activation; memory-efficient but prone to vanishing gradients.
2. **Bidirectional LSTM (BLSTM)**: Uses input, forget, output, and cell gates to manage long-term dependencies in both directions.
3. **RNN with Attention**: Augments the RNN with an attention module to "attend" to specific previous characters during generation.

### Quantitative Evaluation
| Model | Parameters | Final Loss | Novelty Rate (%) | Diversity |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | 23,836 | 1.3194 | 50.00% | 1.0000 |
| **RNN + Attention** | 60,316 | 1.3019 | 40.00% | 1.0000 |
| **BLSTM** | 168,988 | 1.1196 | 10.00% | 1.0000 |

### Qualitative Analysis & Samples
- **Realism**: BLSTM produced the most realistic names (e.g., Umashankar, Vidya) but suffered from heavy memorization (10% novelty).
- **Creativity**: Vanilla RNN and Attention models generated plausible "pseudo-names" (e.g., Naveek, Ramakatt).
- **Failure Modes**: 
  - *Vanilla RNN*: Phonetic hallucinations (e.g., Dived).
  - *Attention*: Sequence termination issues (e.g., Addevivi).

---
