import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ANLI evaluation results")
    parser.add_argument("--csv", type=str, default="gemma3_anli_results.csv",
                        help="Path to CSV results file")
    parser.add_argument("--output_dir", type=str, default="anli_analysis",
                        help="Directory to save the analysis plots")
    parser.add_argument("--show_examples", action="store_true",
                        help="Show example predictions (success and failures)")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to show for each category")
    return parser.parse_args()

def load_data(csv_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Load the evaluation results and analysis."""
    df = pd.read_csv(csv_path)
    
    json_path = csv_path.replace(".csv", "_analysis.json")
    try:
        with open(json_path, 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError:
        print(f"Analysis file {json_path} not found. Please run anli_eval.py first.")
        exit(1)
        
    return df, analysis

def plot_overall_accuracy(analysis: Dict[str, Any], output_dir: str) -> None:
    """Create a bar chart showing overall accuracy and per-class performance."""
    # Extract data
    accuracy = analysis['accuracy']
    class_metrics = analysis['class_metrics']
    random_baseline = analysis.get('random_baseline', 1/3)
    majority_baseline = analysis.get('majority_baseline', 0.5)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Overall accuracy bar
    ax.bar(['Overall Accuracy'], [accuracy], color='blue', alpha=0.7)
    
    # Per-class F1 scores
    classes = list(class_metrics.keys())
    f1_scores = [class_metrics[c]['f1'] for c in classes]
    
    # Plot per-class F1 scores
    positions = range(1, len(classes) + 1)
    ax.bar([p + 0.5 for p in positions], f1_scores, color='green', alpha=0.7)
    
    # Add baselines
    ax.axhline(y=random_baseline, color='red', linestyle='--', label=f'Random Baseline ({random_baseline:.2f})')
    ax.axhline(y=majority_baseline, color='orange', linestyle='--', label=f'Majority Baseline ({majority_baseline:.2f})')
    
    # Add labels
    ax.set_xticks([0] + [p + 0.5 for p in positions])
    ax.set_xticklabels(['Overall Accuracy'] + [f"{c} (F1)" for c in classes])
    
    # Add value labels
    ax.text(0, accuracy + 0.02, f'{accuracy:.2f}', ha='center')
    for i, f1 in enumerate(f1_scores):
        ax.text(i + 1.5, f1 + 0.02, f'{f1:.2f}', ha='center')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('ANLI Evaluation Results: Accuracy and F1 Scores')
    ax.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_accuracy.png")
    print(f"Overall accuracy plot saved to {output_dir}/overall_accuracy.png")
    plt.close()

def plot_confusion_matrix(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot a confusion matrix for model predictions."""
    true_labels = results_df["true_label"]
    pred_labels = results_df["predicted_label"]
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot raw confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                yticklabels=['Entailment', 'Neutral', 'Contradiction'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for ANLI Evaluation')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    print(f"Confusion matrix plot saved to {output_dir}/confusion_matrix.png")
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                yticklabels=['Entailment', 'Neutral', 'Contradiction'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix (row-wise)')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png")
    print(f"Normalized confusion matrix plot saved to {output_dir}/confusion_matrix_normalized.png")
    plt.close()

def plot_round_comparison(analysis: Dict[str, Any], output_dir: str) -> None:
    """Plot accuracy comparison across different ANLI rounds."""
    if 'round_metrics' not in analysis or not analysis['round_metrics']:
        print("No round-specific metrics found in the analysis.")
        return
    
    round_metrics = analysis['round_metrics']
    rounds = list(round_metrics.keys())
    accuracies = [round_metrics[r]['accuracy'] for r in rounds]
    examples = [round_metrics[r]['examples'] for r in rounds]
    
    # Create figure with two axes (one for accuracy, one for sample count)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracies
    bars = ax1.bar(rounds, accuracies, color='blue', alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add random baseline
    random_baseline = analysis.get('random_baseline', 1/3)
    ax1.axhline(y=random_baseline, color='red', linestyle='--', label=f'Random Baseline ({random_baseline:.2f})')
    
    # Add sample counts as text above bars
    for i, (acc, ex) in enumerate(zip(accuracies, examples)):
        ax1.text(i, acc + 0.02, f'n={ex}', ha='center')
        ax1.text(i, acc - 0.05, f'{acc:.2f}', ha='center', color='white', fontweight='bold')
    
    ax1.set_title('ANLI Evaluation Results by Round')
    ax1.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/round_comparison.png")
    print(f"Round comparison plot saved to {output_dir}/round_comparison.png")
    plt.close()

def plot_class_metrics(analysis: Dict[str, Any], output_dir: str) -> None:
    """Plot precision, recall, and F1 for each class."""
    class_metrics = analysis['class_metrics']
    classes = list(class_metrics.keys())
    
    # Extract metrics
    precision = [class_metrics[c]['precision'] for c in classes]
    recall = [class_metrics[c]['recall'] for c in classes]
    f1 = [class_metrics[c]['f1'] for c in classes]
    
    # Set up plot
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#5DA5DA')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#FAA43A')
    rects3 = ax.bar(x + width, f1, width, label='F1 Score', color='#60BD68')
    
    # Labels and formatting
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 Score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add value annotations
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_metrics.png")
    print(f"Class metrics plot saved to {output_dir}/class_metrics.png")
    plt.close()

def analyze_response_quality(results_df: pd.DataFrame, output_dir: str) -> None:
    """Analyze the quality of model responses."""
    # Extract response lengths
    results_df['response_length'] = results_df['generated_response'].str.len()
    
    # Analyze length by correctness
    correct_lengths = results_df[results_df['correct']]['response_length']
    incorrect_lengths = results_df[~results_df['correct']]['response_length']
    
    plt.figure(figsize=(10, 6))
    plt.hist([correct_lengths, incorrect_lengths], bins=20, 
             alpha=0.7, label=['Correct', 'Incorrect'])
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Count')
    plt.title('Response Length by Correctness')
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_quality.png")
    print(f"Response quality plot saved to {output_dir}/response_quality.png")
    plt.close()
    
    # Check if there are patterns in responses
    common_patterns = {}
    for label in [0, 1, 2]:
        label_name = {0: "entailment", 1: "neutral", 2: "contradiction"}[label]
        # Get the most common first words in responses for each predicted class
        pred_responses = results_df[results_df['predicted_label'] == label]['generated_response']
        if len(pred_responses) > 0:
            first_words = [r.split()[0].lower() if len(r.split()) > 0 else "" for r in pred_responses]
            word_counts = pd.Series(first_words).value_counts().head(3).to_dict()
            common_patterns[label_name] = word_counts
    
    # Save the patterns analysis
    with open(f"{output_dir}/response_patterns.json", 'w') as f:
        json.dump(common_patterns, f, indent=2)
    print(f"Response patterns saved to {output_dir}/response_patterns.json")

def display_examples(results_df: pd.DataFrame, num_examples: int = 3) -> None:
    """Display examples of correct and incorrect predictions for each label."""
    label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    print("\n\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)
    
    # Correct predictions
    print("\nCORRECT PREDICTIONS:")
    for label, label_name in label_names.items():
        correct_df = results_df[(results_df["true_label"] == label) & (results_df["correct"] == True)]
        
        if len(correct_df) > 0:
            print(f"\n{label_name.upper()} (correctly classified):")
            examples = correct_df.sample(min(num_examples, len(correct_df)))
            
            for i, (_, example) in enumerate(examples.iterrows()):
                print(f"\nExample {i+1}:")
                print(f"Premise: {example['premise'][:100]}{'...' if len(example['premise']) > 100 else ''}")
                print(f"Hypothesis: {example['hypothesis']}")
                print(f"Model's response: {example['generated_response'][:100]}{'...' if len(example['generated_response']) > 100 else ''}")
        else:
            print(f"\n{label_name.upper()} (correctly classified): No examples found")
    
    # Incorrect predictions
    print("\n\nINCORRECT PREDICTIONS:")
    for label, label_name in label_names.items():
        incorrect_df = results_df[(results_df["true_label"] == label) & (results_df["correct"] == False)]
        
        if len(incorrect_df) > 0:
            print(f"\n{label_name.upper()} (incorrectly classified):")
            examples = incorrect_df.sample(min(num_examples, len(incorrect_df)))
            
            for i, (_, example) in enumerate(examples.iterrows()):
                print(f"\nExample {i+1}:")
                print(f"Premise: {example['premise'][:100]}{'...' if len(example['premise']) > 100 else ''}")
                print(f"Hypothesis: {example['hypothesis']}")
                print(f"True label: {label_name}")
                print(f"Predicted label: {label_names[example['predicted_label']]}")
                print(f"Model's response: {example['generated_response'][:100]}{'...' if len(example['generated_response']) > 100 else ''}")
        else:
            print(f"\n{label_name.upper()} (incorrectly classified): No examples found")

def print_summary(analysis: Dict[str, Any]) -> None:
    """Print overall analysis summary to console."""
    print("\n" + "="*80)
    print("ANLI EVALUATION SUMMARY")
    print("="*80)
    
    # Overall metrics
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Correct predictions: {analysis['correct_count']} ({analysis['accuracy']*100:.1f}%)")
    
    # Baseline comparison
    random_baseline = analysis.get('random_baseline', 1/3)
    majority_baseline = analysis.get('majority_baseline', 0.5)
    print(f"Baselines: Random {random_baseline*100:.1f}%, Majority {majority_baseline*100:.1f}%")
    
    # Per-class metrics
    print("\nClass-wise Performance:")
    for class_name, metrics in analysis['class_metrics'].items():
        print(f"\n  {class_name.upper()}:")
        print(f"    Support: {metrics['support']} examples")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1 Score: {metrics['f1']:.3f}")
    
    # Round-specific metrics, if available
    if 'round_metrics' in analysis and analysis['round_metrics']:
        print("\nPerformance by Round:")
        for round_name, metrics in analysis['round_metrics'].items():
            print(f"\n  {round_name.upper()}:")
            print(f"    Examples: {metrics['examples']}")
            print(f"    Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    
    print("\n" + "="*80)

def main():
    args = parse_args()
    df, analysis = load_data(args.csv)
    
    # Print summary to console
    print_summary(analysis)
    
    # Generate visualizations
    plot_overall_accuracy(analysis, args.output_dir)
    plot_confusion_matrix(df, args.output_dir)
    plot_round_comparison(analysis, args.output_dir)
    plot_class_metrics(analysis, args.output_dir)
    analyze_response_quality(df, args.output_dir)
    
    # Display examples if requested
    if args.show_examples:
        display_examples(df, args.num_examples)

if __name__ == "__main__":
    main()