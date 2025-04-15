# Responsible AI Benchmarking

**Evaluating Gemma LLMs on a variety of Responsible AI Benchmarks**

---

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Results & Analysis](#results--analysis)
- [Acknowledgements](#acknowledgements)
- [To Be Completed](#to-be-completed)

---

## Overview

This repository contains a fully reproducible, open‑source evaluation pipeline for the Gemma family of language models. We run through multiple Responsible AI benchmarks covering fairness and robustness. These benchmarks, to date, have not been published for Gemma3 with 1b parameters. The goal is to provide transparent, shareable results and an extensible framework for future audits.

---

## Project Goals

Assess Gemma LLMs on:
- **Fairness:** Evaluate bias and stereotype trends
- **Robustness:** Test model behavior under adversarial prompts
- **Misinformation:** Assess factual accuracy and resistance to hallucination
- **Interpretability:** Measure explanation quality and chain-of-thought performance

---

## Repository Structure

```
responsible-ai-benchmarking/
    ├── README.MD
    ├── src # Main code repository
        └── gemma3-1b/     # Gemma3 1B folder
            ├── anli/      # ANLI evaluation 
            ├── stereoset/ # StereoSet evaluation
            ├── install_gemma3.py # Helper function to install Gemma3 1b
```
---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DevShel/responsible-ai-benchmarking
   ```

2. **Refer to the specific folder README:**  
   Detailed setup, installation, and benchmark instructions (e.g., for Gemma3 1B StereoSet) are provided in each model folder (e.g., `src/gemma3-1b/README.md`).
---

## Benchmarks & Evaluation
Below are the current benchmarks in this repository, categorized by their Responsible AI domain:

### Fairness

- **StereoSet**: Measures stereotypical biases in language models across gender, race, religion, and profession domains.
  - **Metrics**: Language Modeling Score (LMS), Stereotype Score (SS), ICAT Score

### Robustness

- **Adversarial NLI (ANLI)**: Tests natural language inference capabilities on challenging examples designed to fool state-of-the-art models.
  - **Metrics**: Accuracy, Precision, Recall, F1 Score across three progressively harder rounds

*More benchmarks will be added soon*

---

## Results & Analysis

Below are the results of 3 500-example benchmarks using the code in this repository.

### Gemma 3 1b Benchmark Results

#### StereoSet Intersentence Evaluation (500 examples)

The intersentence test evaluates discourse-level bias by examining how the model continues text containing demographic terms.

**Overall Performance:**
- **Language Modeling Score**: 0.83 (strong content discrimination ability)
- **Stereotype Score**: 0.47 (slight anti-stereotype preference; 0.50 is ideal)
- **ICAT Score**: 78.40 (good balance of language modeling and fairness)

**Domain-Specific Findings:**
- **Gender**: 0.52 stereotype score (nearly balanced), 81.63 ICAT score (highest)
- **Profession**: 0.54 stereotype score (moderate stereotype preference)
- **Race**: 0.42 stereotype score (notable anti-stereotype tendency)
- **Religion**: 0.40 stereotype score (strongest anti-stereotype preference)

#### StereoSet Intrasentence Evaluation (500 examples)

The intrasentence test evaluates sentence-level bias through fill-in-the-blank style completions.

**Overall Performance:**
- **Language Modeling Score**: 0.93 (excellent content discrimination)
- **Stereotype Score**: 0.58 (moderate stereotype preference)
- **ICAT Score**: 77.60 (good overall balance)

**Domain-Specific Findings:**
- **Gender**: 0.73 stereotype score (strongest stereotype preference)
- **Profession**: 0.66 stereotype score (strong stereotype preference)
- **Race**: 0.47 stereotype score (slight anti-stereotype preference)
- **Religion**: 0.60 stereotype score (moderate stereotype preference)

**Key Insights:**
- The model shows different bias patterns between sentence completion (intrasentence) and discourse continuation (intersentence)
- Gender domain shows the most striking difference: balanced in discourse but heavily stereotyped in sentence completion
- Race consistently shows anti-stereotype tendencies in both evaluations
- Stronger language modeling correlates with increased stereotypical associations

#### ANLI Results (500 examples)

The Adversarial NLI benchmark tests natural language inference with examples designed to challenge models.

**Overall Performance:**
- **Accuracy**: 33.6% (168/500 correct predictions)
- Performance is slightly above random chance (33.3%)
- Below majority class baseline (35.4%)

**Class-Specific Metrics:**
- **Entailment**: Precision: 0.33, Recall: 0.60, F1: 0.43
  - High recall but low precision shows bias toward predicting entailment
- **Neutral**: Precision: 0.38, Recall: 0.17, F1: 0.24
  - Very poor recall for neutral relationships
- **Contradiction**: Precision: 0.33, Recall: 0.24, F1: 0.28
  - General difficulty with contradiction

**Performance by Round:**
- **Round 1**: 27.1% accuracy (177 examples)
- **Round 2**: 36.9% accuracy (157 examples)
- **Round 3**: 37.3% accuracy (166 examples)

**Key Insights:**
- Counterintuitively, performance improves in harder rounds
- Strong entailment bias limits overall performance
- Struggles most with neutral relationship detection
- Results align with expectations for a 1B parameter model on this challenging benchmark

---

## Acknowledgements

I gratefully acknowledge the contributions of the StereoSet paper and its authors, whose work has provided a foundational framework for bias evaluation in language models. I also acknowledge the ANLI paper authors for developing a robust benchmark for testing natural language understanding under adversarial conditions.

---

## To Be Completed

- Explore existing benchmarks with a variety of Gemma3 parameter size beyond 1b
- Gemma benchmarking code for other modern Responsible AI benchmark datasets