# Responsible AI Benchmarking

**Evaluating Gemma LLMs on Fairness, Robustness, Misinformation, and Interpretability Benchmarks**

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

This repository contains a fully reproducible, open‑source evaluation pipeline for the Gemma family of language models. We run the models through a curated suite of Responsible‑AI benchmarks—covering fairness, toxicity, privacy, and more—that, to date, have not been published for Gemma. The goal is to provide transparent, shareable results and an extensible framework for future audits.

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
        └── gemma3-1b/    # Gemma3 1B folder
            ├── [Files to run gemma3-1b evaluations]
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

### Fairness

- **StereoSet**: Measures stereotypical biases in language models across gender, race, religion, and profession domains.
  - **Metrics**: Language Modeling Score (LMS), Stereotype Score (SS), ICAT Score

*More benchmarks will be added soon*

---

## Results & Analysis

### Gemma3 1B StereoSet Evaluation

The Gemma3 1B model demonstrates balanced performance on the StereoSet bias benchmark with 200 examples evaluated. The analysis reveals a nearly perfect overall Stereotype Score of 0.497 (with 0.5 being ideal), indicating minimal overall bias in the model's preferences between stereotypical and anti-stereotypical statements. 

The model achieved a strong Language Modeling Score of 0.815, showing its ability to distinguish contextually relevant completions from unrelated ones. This resulted in a high ICAT Score of 81.0, reflecting the model's ability to maintain language understanding while minimizing bias.

When examining bias across different domains:
- **Religion** showed the strongest anti-stereotype tendency (0.429)
- **Race** demonstrated slight anti-stereotype preference (0.462)
- **Profession** exhibited mild stereotype bias (0.545)
- **Gender** displayed the strongest stereotype bias (0.563)

The results suggest that while Gemma3 1B performs remarkably well overall, targeted bias mitigation efforts could focus on gender and profession domains where stereotype preferences are more pronounced. The model's balanced performance across the religion and race categories indicates existing bias mitigation techniques are effective in these areas.

---

## Acknowledgements

I gratefully acknowledge the contributions of the StereoSet paper and its authors, whose work has provided a foundational framework for bias evaluation in language models.

---

## To Be Completed

- More model evaluations
- Extended documentation.
