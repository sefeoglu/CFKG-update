## Continual Factual Knowledge Graph Updates

<p align="center">
  <img src="https://github.com/sefeoglu/CFKG-update/blob/master/figs/pipeline_method.png" width="800"/>
</p>

# SHACL-Based Symbolic Memory for Continual Relation Extraction

This repository contains the implementation of a framework combining **continual few-shot relation extraction** with **schema-aware symbolic validation**. The approach leverages synaptic regularization methods and a SHACL-based symbolic memory layer to mitigate catastrophic forgetting and distinguish between true hallucinations and semantically valid out-of-schema predictions.

## Overview

Large Language Models (LLMs) used for Knowledge Graph (KG) construction often suffer from **catastrophic forgetting** in dynamic environment. Furthermore, they frequently generate relations that, while semantically valid in an open-world scenario, are treated as errors because they fall outside predefined schemas. 

This project addresses these challenges through:
1.  **Synaptic Regularization**: Utilizing EWC, SI, and MAS to maintain model stability during incremental learning.
2.  **Symbolic Memory**: A SHACL-based validation layer that captures canonical and inverse relations.
3.  **LLM-as-a-Judge**: A mechanism for uncertainty assessment and explanation, determining when human validation is required.

---

## 🛠️ Framework Structure

* The pipeline consists of four primary stages
* **Continual LLM Tuning**: Fine-tuning (specifically Flan-T5 Base) using synaptic approaches to preserve important parameters
* **Canonical & Inverse Relation Discovery**: Identifying semantically close predictions using Wikidata and human expertise
* **SHACL Validation**: Using Shapes Constraint Language to interpret LLM-generated relations against known schemas
* **Explainability**: Generating natural-language rationales and uncertainty scores (1-10) via an LLM-as-a-Judge (e.g., Gemini)

---

## 📊 Key Features & Results

### Synaptic Regularization Methods
The framework implements three bio-inspired regularizers:
| Method | Description |
| :--- | :--- |
| **EWC** | Elastic Weight Consolidation; uses the Fisher Information Matrix to identify important parameters.|
| **SI** | Synaptic Intelligence; computes parameter importance online as a structural regularizer. |
| **MAS** | Memory-Aware Synapses; based on Hebbian learning to keep weights close to previous training. |

### Performance on Benchmarks
Experiments were conducted on **TACRED** and **FewRel** datasets:
* **TACRED**: Synaptic approaches consistently outperformed baselines, achieving up to **90.86%** accuracy at the end of incremental tuning.
* **Sample Selection**: K-means clustering with **Euclidean distance** was found to be the most effective strategy for selecting diverse training shots.
* **Hallucination Resolution**: The model using **SI** was found to be less prone to hallucinations across both datasets.

---

## 📂 Repository Content

* `src/bio-inspired_regularizers/`: Implementations of `ewc.py`, `si.py`, and `mas.py`.
* `SHACL Shapes`: Symbolic memory constraints for canonical and inverse relation detection.
* `LLM-as-a-Judge`: Templates for base and textual entailment-based evaluation formats.

---

## 📝 Citation

## References


