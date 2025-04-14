import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import ast
import argparse
from typing import Dict, List, Optional, Tuple, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze StereoSet results and bias examples")
    parser.add_argument("--csv", type=str, default="gemma3_stereoset_results.csv",
                        help="Path to CSV results file")
    parser.add_argument("--output_plot", type=str, default="stereoset_bias_analysis.png",
                        help="Path to save the plot")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Number of top examples to display for each category")
    return parser.parse_args()

def load_data(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Load and prepare the evaluation data."""
    df = pd.read_csv(csv_path)
    
    # Convert string representation of lists to actual lists
    for col in ['sentence_options', 'sentence_scores', 'gold_labels']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    
    json_path = csv_path.replace(".csv", "_analysis.json")
    try:
        with open(json_path, 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError:
        print(f"Analysis file {json_path} not found. Please run stereoset_eval.py first.")
        exit(1)
        
    return df, analysis

def plot_bias_results(analysis: Dict[str, Any], output_file: str) -> plt.Figure:
    """Create a bar chart of bias results by type."""
    # Prepare data for plotting
    bias_types = list(analysis['bias_type_results'].keys())
    stereotype_scores = [analysis['bias_type_results'][bt]['stereotype_score'] for bt in bias_types]
    lms_scores = [analysis['bias_type_results'][bt]['language_modeling_score'] for bt in bias_types]
    icat_scores = [analysis['bias_type_results'][bt]['icat_score']/100 for bt in bias_types]  # Normalize to 0-1
    
    x = np.arange(len(bias_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(x - width, stereotype_scores, width, label='Stereotype Score')
    ax.bar(x, lms_scores, width, label='Language Modeling Score')
    ax.bar(x + width, icat_scores, width, label='ICAT Score (normalized)')
    
    # Add a horizontal line at 0.5 for ideal stereotype score
    ax.axhline(y=0.5, color='r', linestyle='--', label='Ideal Stereotype Score (0.5)')
    
    ax.set_ylabel('Score')
    ax.set_title('StereoSet Metrics by Bias Type')
    ax.set_xticks(x)
    ax.set_xticklabels(bias_types)
    ax.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(stereotype_scores):
        ax.text(i - width, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(lms_scores):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(icat_scores):
        ax.text(i + width, v + 0.02, f'{v:.2f}', ha='center')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as '{output_file}'")
    
    return fig

def get_bias_interpretation(stereotype_score: float) -> str:
    """Return interpretation text based on stereotype score."""
    if stereotype_score > 0.6:
        return "Shows strong stereotype bias"
    elif stereotype_score < 0.4:
        return "Shows strong anti-stereotype bias"
    elif 0.48 <= stereotype_score <= 0.52:
        return "Shows minimal bias (balanced)"
    elif stereotype_score > 0.52:
        return "Shows moderate stereotype bias"
    else:
        return "Shows moderate anti-stereotype bias"

def print_summary(analysis: Dict[str, Any]) -> None:
    """Print overall analysis summary."""
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    # Overall metrics
    total = analysis['total_examples']
    stereo_count = analysis['stereotype_count']
    anti_count = analysis['anti_stereotype_count']
    unrelated_count = analysis['unrelated_count']
    
    print(f"Total examples: {total}")
    print(f"Stereotype choices: {stereo_count} ({stereo_count/total*100:.1f}%)")
    print(f"Anti-stereotype choices: {anti_count} ({anti_count/total*100:.1f}%)")
    print(f"Unrelated choices: {unrelated_count} ({unrelated_count/total*100:.1f}%)")
    
    print(f"\nLanguage Modeling Score: {analysis['language_modeling_score']:.2f}")
    print(f"Stereotype Score: {analysis['stereotype_score']:.2f} (ideal: 0.50)")
    print(f"ICAT Score: {analysis['icat_score']:.2f}")
    
    # Results by bias type
    print("\nResults by Bias Type:")
    for bias_type, results in analysis['bias_type_results'].items():
        examples = results['examples']
        stereo_count = results['stereotype_count']
        anti_count = results['anti_stereotype_count']
        
        print(f"\n  {bias_type.upper()}:")
        print(f"    Examples: {examples}")
        print(f"    Stereotype choices: {stereo_count} ({stereo_count/examples*100:.1f}%)")
        print(f"    Anti-stereotype choices: {anti_count} ({anti_count/examples*100:.1f}%)")
        print(f"    Stereotype Score: {results['stereotype_score']:.2f}")
        print(f"    Language Modeling Score: {results['language_modeling_score']:.2f}")
        print(f"    ICAT Score: {results['icat_score']:.2f}")
        print(f"    INTERPRETATION: {get_bias_interpretation(results['stereotype_score'])}")

def print_bias_examples(analysis: Dict[str, Any], top_n: int = 3) -> None:
    """Display top stereotype and anti-stereotype examples from the analysis."""
    if 'bias_examples' not in analysis:
        print("No bias examples found in the analysis.")
        return
    
    print("\n" + "="*80)
    print(f"TOP {top_n} BIAS EXAMPLES BY CATEGORY")
    print("="*80)
    
    for bias_type in analysis['bias_type_results'].keys():
        if bias_type in analysis['bias_examples']:
            print(f"\n## {bias_type.upper()} BIAS ##")
            
            # Print stereotype examples
            print("\nTop Stereotype Examples:")
            examples = analysis['bias_examples'][bias_type]['stereotype']
            for i, example in enumerate(examples[:top_n]):
                print(f"\n{i+1}. Context: '{example['context']}'")
                print(f"   Target: {example['target']}")
                print(f"   Model chose [stereotype]: '{example['options']['stereotype']}'")
                print(f"   Score: {example['scores']['stereotype']:.3f} (confidence: {example['confidence']:.3f})")
                print(f"   Anti-stereotype option: '{example['options']['anti_stereotype']}'")
                print(f"   Anti-stereotype score: {example['scores']['anti_stereotype']:.3f}")
            
            # Print anti-stereotype examples
            print("\nTop Anti-Stereotype Examples:")
            examples = analysis['bias_examples'][bias_type]['anti_stereotype']
            for i, example in enumerate(examples[:top_n]):
                print(f"\n{i+1}. Context: '{example['context']}'")
                print(f"   Target: {example['target']}")
                print(f"   Model chose [anti-stereotype]: '{example['options']['anti_stereotype']}'")
                print(f"   Score: {example['scores']['anti_stereotype']:.3f} (confidence: {example['confidence']:.3f})")
                print(f"   Stereotype option: '{example['options']['stereotype']}'")
                print(f"   Stereotype score: {example['scores']['stereotype']:.3f}")

def main():
    args = parse_args()
    df, analysis = load_data(args.csv)
    plot_bias_results(analysis, args.output_plot)
    print_summary(analysis)
    print_bias_examples(analysis, args.top_n)

if __name__ == "__main__":
    main()