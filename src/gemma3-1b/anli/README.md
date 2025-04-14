# Gemma 3 1b: ANLI Evaluation

This repository contains code to evaluate the Gemma 3 1b model on the Adversarial Natural Language Inference (ANLI) dataset:

1.  **Evaluation:** Test the model on natural language inference across three rounds of increasingly difficult examples.
2.  **Analysis:** Calculate accuracy, precision, recall, and F1 scores for the evaluation.
3.  **Visualization:** Generate insightful plots to understand model performance.

## Folder Structure

```
gemma3-1b/anli/
├── anli_eval.py           # Script to evaluate Gemma3 on ANLI dataset
├── analyze_anli_results.py   # Script to analyze and visualize results
├── requirements.txt       # Required Python packages
├── example_anli_results_analysis.txt # Sample analysis output
└── README.md              # This documentation file
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

## Understanding ANLI

The Adversarial NLI (ANLI) dataset was created to be challenging for models. It was collected using a human-and-model-in-the-loop process over three rounds, with each round designed to be harder than the last.

Each example contains:
- A `premise` statement
- A `hypothesis` statement
- A `label` indicating the relationship:
  - `0` = entailment (hypothesis follows from premise)
  - `1` = neutral (hypothesis may or may not follow)
  - `2` = contradiction (hypothesis contradicts premise)

## Evaluation Approach: Prompting Gemma 3

This project evaluates Gemma 3, a **decoder-only** transformer model, on the ANLI task. Because Gemma 3 is a generative model (unlike the encoder models like BERT/RoBERTa used in the original ANLI paper), our evaluation method uses prompting:

1.  **Input:** We format the premise and hypothesis into a prompt using Gemma 3's chat template. This includes specific control tokens like `[BOS]`, `<start_of_turn>`, and `<end_of_turn>` required by the instruction-tuned model. Few-shot examples can optionally be included in the prompt.
2.  **Inference:** We perform a single, deterministic generation step using `model.generate` with `do_sample=False` and `temperature=0.0`. This aligns with the deterministic nature of the original ANLI evaluation.
3.  **Output & Classification:** The model generates text completion in response to the prompt. We then parse this generated text (e.g., looking for keywords like "entailment", "neutral", "contradiction") to extract the predicted label.

### Differences from Original ANLI Evaluation

-   **Model Type & Fine-tuning:** The original ANLI paper evaluated models (BERT, RoBERTa) that were *fine-tuned* specifically as 3-way classifiers, typically using the `[CLS]` token representation fed into a softmax layer. We are evaluating a pre-trained/instruction-tuned *decoder-only* model (Gemma 3) using its generative capabilities via *prompting*, without specific NLI fine-tuning for this codebase.
-   **Classification Mechanism:** Original ANLI used direct softmax output over the 3 classes. We infer the class by *parsing the generated text*.

**Why the Difference?** This approach is necessary because Gemma 3 is a decoder-only model without a readily available classification head for NLI in its standard instruction-tuned form. Prompting allows us to assess the model's NLI understanding using its inherent instruction-following and in-context learning capabilities.

## Running ANLI Evaluation

The evaluation script automatically downloads the ANLI dataset from Hugging Face.

```bash
# Evaluate on ANLI Round 1 test set
python anli_eval.py --round 1 --split test

# Evaluate on all rounds, with all possible samples
python anli_eval.py --round all --split test

# For faster evaluation with quantization (ensure model was installed with --quantize)
python anli_eval.py --round 2 --split test --quantize
```

### ANLI Evaluation Options

- `--round`: ANLI round to evaluate (`1`, `2`, `3`, or `all`)
- `--split`: Dataset split to use (`train`, `dev`, `test`)
- `--max_samples`: Limit number of evaluation samples
- `--device`: Device to use (`cuda` or `cpu`)
- `--quantize`: Use 4-bit quantization for faster/lower-memory evaluation
- `--few_shot`: Use few-shot examples in prompt (default: True)
- `--cot`: Use chain-of-thought reasoning in prompt (default: False)

### Output Files

- CSV file with per-example results (`gemma3_anli_results.csv`)
- JSON file with aggregate metrics (`gemma3_anli_results_analysis.json`)

## Analyzing Results

Run the analysis script to visualize and interpret the results:

```bash
# Basic analysis
python analyze_anli_results.py

# Show example predictions
python analyze_anli_results.py --show_examples --num_examples 5
```

<span style="color:crimson">To see an example of an analysis run, see "./example_anli_results_analysis.txt"</span>

### Analysis Options

- `--csv`: Path to CSV results file (default: `gemma3_anli_results.csv`)
- `--output_dir`: Directory to save visualizations (default: `anli_analysis`)
- `--show_examples`: Display example predictions (successes and failures)
- `--num_examples`: Number of examples to show per category (default: 3)

### Generated Visualizations

- Overall accuracy and F1 scores by class
- Confusion matrix
- Performance comparison across ANLI rounds
- Precision, recall, and F1 scores for each class
- Analysis of response characteristics

## Interpretation of Metrics

- **Accuracy**: Percentage of examples correctly classified
- **Precision**: Ratio of true positives to all positive predictions for each class
- **Recall**: Ratio of true positives to all actual positives for each class
- **F1 Score**: Harmonic mean of precision and recall (balanced measure)

Results are provided for individual rounds and the overall dataset to assess model performance across different difficulty levels.

## Troubleshooting

- **Model Download Issues**: Ensure Hugging Face authentication and model license acceptance
- **Dependency Issues**: Use Python 3.10 and install all requirements
- **Memory Issues**: Use the `--quantize` option with larger models or limited GPU memory
- **Dataset Loading Errors**: Check internet connection and Hugging Face API status

## Citation

```bibtex
@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and
      Williams, Adina  and
      Dinan, Emily  and
      Bansal, Mohit  and
      Weston, Jason  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## License

Ensure you comply with the Gemma 3 1b model license as stated on its [Hugging Face page](https://huggingface.co/google/gemma-3-1b-it) and the ANLI dataset license (CC BY-NC 4.0).