# Gemma3 1b: Installation and StereoSet Evaluation

This repository contains code to work with the Gemma3 1b model for bias evaluation:

1. **Installation:** Download and locally install the Gemma3 1b model and tokenizer.
2. **StereoSet Evaluation:** Measure stereotypical biases across gender, race, religion, and profession domains.
3. **Results Analysis:** Analyze and visualize bias patterns with detailed examples.

## Folder Structure

```
Gemma3-1b/
├── install_gemma3.py       # Script to download and locally save the model
├── stereoset_eval.py       # Script to evaluate bias on the StereoSet dataset
├── analyze_results.py      # Script to analyze and visualize results
├── requirements.txt        # Required Python packages
├── example_results_analysis.txt        # Sample run of "analyze_results.py" on a "stereoset_eval.py" run
└── README.md               # This documentation file
```

## Prerequisites

- Python 3.10 (recommended)
- pip (installed with Python)
- (Optional) NVIDIA GPU with CUDA drivers
- Hugging Face CLI installed and configured (`huggingface-cli login`)
- Accepted the Gemma3 1b model terms on [Hugging Face](https://huggingface.co/google/gemma-3-1b-it)

## 1. Gemma3 1b Installation

### Quick Setup

```bash
# Create and activate environment
python3.10 -m venv env_py310
#MacOS/Linux:
source env_py310/bin/activate 

# On Windows: 
env_py310\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# ! Make sure to visit the model page and click "Acknowledge License" and follow the steps to accept the terms of use.
Model Link: https://huggingface.co/google/gemma-3-1b-it

# Download and save model locally
python install_gemma3.py  # Add --quantize for 4-bit quantization
```

The model and tokenizer are saved in the `gemma3_model` directory by default.

## 2. StereoSet Evaluation

The evaluation script automatically downloads the original StereoSet dataset from the paper authors' GitHub repository.

```bash
python stereoset_eval.py --subset intersentence --max_samples=30
```

### StereoSet Evaluation Option Flags
- `--subset`: Dataset subset (`intersentence` or `intrasentence`)
- `--max_samples`: Limit evaluation samples
- `--num_examples`: Number of bias examples to save (default: 5)
- `--device`: Device to use (`cuda` or `cpu`)
- `--quantize`: Use 4-bit quantization

### Output Files

- CSV file with per-example results (`gemma3_stereoset_results.csv`)
- JSON file with aggregate metrics and bias examples (`gemma3_stereoset_results_analysis.json`)

## 3. Analyzing Results

Run the analysis script to visualize and interpret the results:

```bash
python analyze_results.py --top_n 3
```

To see an example of an analysis run, see "./example_results_analysis.txt"

### Result Analysis Option Flags

- `--csv`: Path to CSV results file (default: `gemma3_stereoset_results.csv`)
- `--output_plot`: Path to save visualization (default: `stereoset_bias_analysis.png`)
- `--top_n`: Number of top examples to display


### Understanding Metrics

- **Language Modeling Score (LMS)**: Measures the model's language understanding (0-1)
- **Stereotype Score (SS)**: Measures stereotype bias (0.5 is ideal, >0.5 shows stereotype bias)
- **ICAT Score**: Combined metric balancing language understanding and bias avoidance (0-100)

### How the Model Selects Completions

The StereoSet evaluation doesn't explicitly ask the model to choose between sentences. Instead:

1. **Scoring:** The model calculates a log probability score for each completion option
2. **Selection:** The completion with the highest score (lowest perplexity) is considered the model's "choice"
3. **Analysis:** This choice is then mapped to its corresponding category (stereotype, anti-stereotype, or unrelated)

This approach measures implicit bias by examining which types of statements the model finds most probable, without prompting it to make conscious choices between stereotypical and non-stereotypical content.

### What You'll Get

1. **Visual bias analysis chart** showing metrics by category (stereoset_bias_analysis.png)
2. **Summary statistics** with overall results and breakdowns by bias type
3. **Top bias examples** sorted by confidence for each category
4. **Interpretations** of bias patterns and their severity

## Troubleshooting

- **Model Download Issues**: Ensure Hugging Face authentication and model license acceptance
- **Dependency Issues**: Use Python 3.10 and install all requirements (ideally using the virtualenv in the instructions for dependency management)

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

Ensure you comply with the Gemma3 1b model license as stated on its [Hugging Face page](https://huggingface.co/google/gemma-3-1b-it).