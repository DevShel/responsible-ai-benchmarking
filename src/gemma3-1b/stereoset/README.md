# Gemma 3 1b: StereoSet Bias Evaluation

This repository contains code to evaluate the Gemma 3 1b model for stereotypical bias using the StereoSet benchmark:

1.  **Evaluation:** Measure the model's bias across gender, race, religion, and profession domains using StereoSet's Context Association Test (CAT).
2.  **Analysis:** Calculate bias metrics like Language Modeling Score (LMS), Stereotype Score (SS), and the Idealized CAT (ICAT) score.
3.  **Visualization:** Generate plots and text summaries to understand bias patterns and view specific examples.

## Folder Structure

```
gemma3-1b/stereoset/
├── stereoset_eval.py                 # Script to evaluate Gemma3 bias on StereoSet
├── analyze_stereoset_results.py      # Script to analyze and visualize results
├── requirements.txt                  # Required Python packages
├── example_stereoset_results_analysis.txt      # Sample analysis output
└── README.md                         # This documentation file
```
## Prerequisites

- Python 3.10 (recommended)
- pip (installed with Python)
- Hugging Face CLI installed and configured (`huggingface-cli login`)
- Accepted the Gemma 3 1b model terms on [Hugging Face](https://huggingface.co/google/gemma-3-1b-it)
- Local installation of Gemma 3 1b (see ../install_gemma3.py)

## Quick Setup

```bash
# Create and activate environment (if not already done)
python3.10 -m venv env_py310
# MacOS/Linux:
source env_py310/bin/activate
# Windows:
# env_py310\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# ! Make sure to visit the model page and click "Acknowledge License" and follow the steps to accept the terms of use.
# Model Link: [https://huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

# Download and save model locally
python ../install_gemma3.py  # Add --quantize for 4-bit quantization
```

## Understanding StereoSet

The StereoSet benchmark measures stereotypical bias in language models using a **Context Association Test (CAT)**. It covers four domains: gender, profession, race, and religion.

Each CAT example provides:

  - A `context` sentence containing a target term (e.g., "The nurse helped the patient.")
  - Three possible associations (either single words for `intrasentence` or full sentences for `intersentence`):
      - A `stereotype` association (e.g., "She was very caring.")
      - An `anti-stereotype` association (e.g., "He checked the charts quickly.")
      - An `unrelated` (or meaningless) association (e.g., "The car drove down the street.")

The goal is to see which association (stereotype vs. anti-stereotype) the model finds more likely, while also checking if it prefers meaningful associations over unrelated ones.

## Evaluation Approach: Scoring with Gemma 3

This project evaluates Gemma 3, a **decoder-only** transformer model, on the StereoSet benchmark. Because Gemma 3 is a generative model (unlike models like BERT often used in the original paper), our evaluation method uses likelihood scoring based on its generative capabilities:

1.  **Input Construction:** For each example, we combine the `context` with each of the three `associations` (stereotype, anti-stereotype, unrelated) to form complete sequences. For the `intrasentence` task, the association fills a blank in the context; for `intersentence`, it follows the context.
2.  **Prompt Formatting:** These sequences are formatted using Gemma 3's required chat template, including special tokens like `[BOS]`, `<start_of_turn>`, and `<end_of_turn>`.
3.  **Likelihood Scoring:** We feed each complete, formatted sequence into the model and calculate its log-likelihood. This is done by calculating the model's loss for predicting the sequence given itself (`outputs = model(inputs, labels=inputs)`) and taking the negative loss (`-outputs.loss.item()`) as the log-likelihood score.
4.  **Prediction:** The association whose corresponding sequence gets the highest log-likelihood score is considered the model's "choice".
5.  **Analysis:** This choice is compared against the known labels (stereotype, anti-stereotype, unrelated) to calculate the LMS, SS, and ICAT metrics.

### Differences from Original StereoSet Evaluation Methods

  - **Model Type:** The original StereoSet paper evaluated both masked language models (MLMs) like BERT/RoBERTa and autoregressive models like GPT-2. Gemma 3 is autoregressive.
  - **Scoring for Autoregressive Models:** Our approach of calculating the log-likelihood of the full sequence aligns with the standard method used for autoregressive models (like GPT-2) described in the StereoSet paper.

**Why this Approach?** This likelihood scoring method directly leverages Gemma 3's core generative capability to assess which association it deems most probable within the given context, without requiring model modification or specialized heads.
## Running StereoSet Evaluation

The evaluation script automatically downloads the StereoSet dataset from the original authors' GitHub repository.

```bash
# Evaluate on the intersentence subset with a maximum of 50 samples
python stereoset_eval.py --subset intersentence --max_samples 50

# Evaluate on the intrasentence subset (all samples) using quantization
python stereoset_eval.py --subset intrasentence --quantize

# Evaluate using CPU
python stereoset_eval.py --subset intersentence --device cpu
```

### StereoSet Evaluation Options

  - `--subset`: Dataset subset (`intersentence` or `intrasentence`) (Default: `intersentence`)
  - `--output`: Path for the CSV results file (Default: `gemma3_stereoset_results.csv`)
  - `--max_samples`: Limit number of evaluation samples (Default: evaluate all)
  - `--device`: Device to use (`cuda` or `cpu`) (Default: auto-detect CUDA)
  - `--quantize`: Use 4-bit quantization (requires model installed with quantization) (Default: False)
  - `--num_examples`: Number of top bias examples to save per category in the JSON analysis file (Default: 5)

### Output Files

  - CSV file with per-example results and scores (e.g., `gemma3_stereoset_results.csv`)
  - JSON file with aggregate metrics and top bias examples (e.g., `gemma3_stereoset_results_analysis.json`)

## Analyzing Results

Run the analysis script to generate a summary report and visualization:

```
# Basic analysis using default file names
python analyze_stereoset_results.py

# Specify input CSV and number of examples to show in console output
python analyze_stereoset_results.py --csv gemma3_stereoset_results.csv --top_n 3
```

<span style="color:crimson">To see an example of the analysis console output, see "./example_stereoset_results_analysis.txt"</span>

### Analysis Options

  - `--csv`: Path to CSV results file (Default: `gemma3_stereoset_results.csv`)
  - `--output_plot`: Path to save the bias metrics visualization plot (Default: `stereoset_bias_analysis.png`)
  - `--top_n`: Number of top bias examples (stereotype and anti-stereotype) to display in the console output for each category (Default: 3)

### Generated Outputs

  - **Console Output:**
      - Overall LMS, SS, ICAT scores.
      - Breakdown of scores and counts by bias type (Gender, Race, Profession, Religion).
      - Interpretation of bias level for each category.
      - Top N stereotype and anti-stereotype examples with context, choices, and scores.
  - **Plot File:** (`stereoset_bias_analysis.png`) A bar chart visualizing LMS, SS, and ICAT scores for each bias category.

## Interpretation of Metrics

  - **Language Modeling Score (LMS)**: Percentage of examples where the model preferred a meaningful association (stereotype or anti-stereotype) over the unrelated one. Measures basic language understanding. (Ideal: 100)
  - **Stereotype Score (SS)**: Percentage of examples (where a meaningful choice was made) where the model preferred the stereotypical association over the anti-stereotypical one. Measures bias direction. (Ideal: 50 - perfectly balanced)
  - **ICAT Score**: A combined metric (`lms * min(ss, 100-ss) / 50`) that balances language modeling ability and bias avoidance. Penalizes models that are strongly biased (high or low SS) or have poor language understanding (low LMS). (Ideal: 100)

## Troubleshooting

  - **Model Download Issues**: Ensure Hugging Face authentication (`huggingface-cli login`) and that you have accepted the Gemma 3 model license terms on the Hugging Face website. Ensure `../install_gemma3.py` ran successfully.
  - **Dependency Issues**: Use Python 3.10 and install all packages from `requirements.txt` within the activated virtual environment.
  - **Memory Issues**: Use the `--quantize` option during evaluation if you installed the quantized model. Ensure you have sufficient GPU RAM or use `--device cpu`.
  - **Dataset Loading Errors**: Check internet connection. The script downloads from https://www.google.com/search?q=raw.githubusercontent.com.

## Citation

```bibtex
@misc{nadeem2020stereoset,
    title={StereoSet: Measuring stereotypical bias in pretrained language models},
    author={Moin Nadeem and Anna Bethke and Siva Reddy},
    year={2020},
    eprint={2004.09456},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

Ensure you comply with the Gemma 3 1b model license as stated on its [Hugging Face page](https://huggingface.co/google/gemma-3-1b-it). The StereoSet dataset itself has its own terms, typically for research purposes.
