import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = "gemma3_model"

def parse_args():
    """Parse command line arguments for ANLI evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Gemma3 1b model on ANLI dataset")
    parser.add_argument("--round", type=str, default="1", choices=["1", "2", "3", "all"],
                        help="ANLI round to evaluate (1, 2, 3, or all)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output", type=str, default="gemma3_anli_results.csv",
                        help="Output file path")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--few_shot", action="store_true", default=True,
                        help="Use few-shot examples in prompt")
    parser.add_argument("--cot", action="store_true", 
                        help="Use chain-of-thought reasoning (for analysis)")
    return parser.parse_args()

def load_anli_dataset(round_num: str, split: str) -> pd.DataFrame:
    """Load the ANLI dataset from Hugging Face for the specified round and split."""
    logger.info(f"Loading ANLI dataset round {round_num}, split {split}")
    
    try:
        dataset_name = "facebook/anli"
        
        if round_num != "all":
            # Load dataset for a specific round
            split_name = f"{split}_r{round_num}"
            dataset = load_dataset(dataset_name, split=split_name)
            
            # Convert to DataFrame
            data = [{
                "uid": item["uid"],
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": item["label"],
                "round": int(round_num)
            } for item in dataset]
            
            df = pd.DataFrame(data)
            
        else:
            # For 'all', load each round separately and concatenate
            combined_data = []
            for r in ["1", "2", "3"]:
                split_name = f"{split}_r{r}"
                dataset = load_dataset(dataset_name, split=split_name)
                
                # Use the actual round number (r) instead of 'all'
                for item in dataset:
                    combined_data.append({
                        "uid": item["uid"],
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label": item["label"],
                        "round": int(r)  # Using the actual round number
                    })
            
            df = pd.DataFrame(combined_data)
        
        logger.info(f"Successfully loaded {len(df)} examples from {'all rounds' if round_num == 'all' else f'round {round_num}'}, split {split}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def get_few_shot_examples():
    """Return few-shot examples for the NLI task."""
    return [
        {
            "premise": "A woman is dancing in a nightclub.",
            "hypothesis": "A woman is having fun at a social gathering.",
            "label": "neutral",
            "explanation": "The premise doesn't specify if the woman is having fun, so this is neutral."
        },
        {
            "premise": "The cat sat on the mat and watched the birds outside.",
            "hypothesis": "The cat was on the mat.",
            "label": "entailment",
            "explanation": "If the cat sat on the mat, then it was definitely on the mat."
        },
        {
            "premise": "The teacher handed out worksheets to the class.",
            "hypothesis": "No learning materials were distributed.",
            "label": "contradiction",
            "explanation": "Worksheets are learning materials, so this contradicts the premise."
        }
    ]

def create_classification_prompt(premise: str, hypothesis: str, use_few_shot: bool = True) -> str:
    """Create a direct classification prompt for NLI."""
    # Start with task description
    prompt = "Task: Natural Language Inference\n\n"
    prompt += "Determine if the hypothesis is entailed by the premise, contradicts the premise, or is neutral to the premise.\n\n"
    
    # Add few-shot examples if requested
    if use_few_shot:
        examples = get_few_shot_examples()
        prompt += "Examples:\n\n"
        
        for ex in examples:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += f"Answer: {ex['label']}\n\n"
    
    # Add current example
    prompt += f"Now classify this pair:\n\n"
    prompt += f"Premise: {premise}\n"
    prompt += f"Hypothesis: {hypothesis}\n\n"
    prompt += "Answer: "
    
    return prompt

def create_cot_prompt(premise: str, hypothesis: str, use_few_shot: bool = True) -> str:
    """Create a chain-of-thought reasoning prompt for NLI."""
    # Start with task description
    prompt = "Task: Natural Language Inference with Reasoning\n\n"
    prompt += "Determine if the hypothesis is entailed by the premise, contradicts the premise, or is neutral to the premise.\n"
    prompt += "Think through your reasoning step by step, then provide the final classification.\n\n"
    
    # Add few-shot examples if requested
    if use_few_shot:
        examples = get_few_shot_examples()
        prompt += "Examples:\n\n"
        
        for ex in examples:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += "Reasoning: " + ex["explanation"] + "\n"
            prompt += f"Answer: {ex['label']}\n\n"
    
    # Add current example
    prompt += f"Now classify this pair:\n\n"
    prompt += f"Premise: {premise}\n"
    prompt += f"Hypothesis: {hypothesis}\n\n"
    prompt += "Reasoning: "
    
    return prompt

def extract_label(text: str) -> int:
    """Extract NLI label from generated text."""
    text = text.lower().strip()
    
    # Check for direct mentions of the label
    if any(word in text for word in ["entailment", "entails", "entail"]):
        return 0
    elif any(word in text for word in ["contradiction", "contradicts", "contradict"]):
        return 2
    elif any(word in text for word in ["neutral"]):
        return 1
    
    # If no direct mention, look for other indicators
    if "yes" in text or "follow" in text or "definitely correct" in text or "true" in text:
        return 0
    elif "no" in text or "not" in text or "false" in text or "incorrect" in text or "wrong" in text:
        return 2
    
    # Default to neutral
    return 1

def evaluate_gemma_on_anli(model, tokenizer, dataset: pd.DataFrame, args) -> pd.DataFrame:
    """Evaluate the model on ANLI using direct classification approach."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")
    
    df = dataset.copy()
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42)
        logger.info(f"Using {args.max_samples} samples for evaluation")
    
    results = []
    model.eval()
    
    # Label mapping
    id_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            premise = row["premise"]
            hypothesis = row["hypothesis"]
            true_label = row["label"]
            
            # Create appropriate prompt
            if args.cot:
                # Chain-of-thought prompt
                prompt = create_cot_prompt(premise, hypothesis, use_few_shot=args.few_shot)
            else:
                # Direct classification prompt
                prompt = create_classification_prompt(premise, hypothesis, use_few_shot=args.few_shot)
            
            # Create messages for chat format
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant skilled at natural language inference tasks."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            ]
            
            # Tokenize
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(device)
            
            try:
                # Generate a classification response
                output_ids = model.generate(
                    inputs,
                    max_new_tokens=100 if args.cot else 10,  # Shorter for direct classification
                    do_sample=False,                         # Deterministic generation
                    temperature=0.0                          # No randomness
                )
                
                # Decode the response
                generated_text = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                
                # Extract the classification
                predicted_label = extract_label(generated_text)
                
                # Store results
                results.append({
                    "uid": row.get("uid", ""),
                    "round": row.get("round", 0),
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "generated_response": generated_text,
                    "correct": predicted_label == true_label
                })
                
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                # Default to neutral on errors
                results.append({
                    "uid": row.get("uid", ""),
                    "round": row.get("round", 0),
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "true_label": true_label,
                    "predicted_label": 1,  # Default to neutral
                    "generated_response": "Error during generation",
                    "correct": 1 == true_label
                })
    
    return pd.DataFrame(results)

def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate metrics from ANLI evaluation results."""
    total_examples = len(results_df)
    correct_count = results_df["correct"].sum()
    
    # Calculate overall accuracy
    accuracy = correct_count / total_examples if total_examples > 0 else 0
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in [0, 1, 2]:
        label_name = {0: "entailment", 1: "neutral", 2: "contradiction"}[label]
        
        # Examples with this true label
        true_positives = results_df[(results_df["true_label"] == label) & 
                                   (results_df["predicted_label"] == label)].shape[0]
        false_negatives = results_df[(results_df["true_label"] == label) & 
                                    (results_df["predicted_label"] != label)].shape[0]
        false_positives = results_df[(results_df["true_label"] != label) & 
                                    (results_df["predicted_label"] == label)].shape[0]
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": results_df[results_df["true_label"] == label].shape[0]
        }
    
    # Calculate results by round if applicable
    round_metrics = {}
    if "round" in results_df.columns:
        for round_num in sorted(results_df["round"].unique()):
            round_df = results_df[results_df["round"] == round_num]
            if len(round_df) > 0:
                round_accuracy = round_df["correct"].sum() / len(round_df)
                
                round_metrics[f"round_{round_num}"] = {
                    "examples": len(round_df),
                    "accuracy": round_accuracy
                }
    
    # Calculate baseline comparison
    random_baseline = 1/3  # For 3-class classification
    majority_baseline = results_df["true_label"].value_counts().max() / total_examples
    
    # Compile the analysis
    analysis = {
        "total_examples": total_examples,
        "correct_count": int(correct_count),
        "accuracy": accuracy,
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "class_metrics": class_metrics,
        "round_metrics": round_metrics
    }
    
    return analysis

def main():
    args = parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    try:
        dataset = load_anli_dataset(args.round, args.split)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Load locally installed model and tokenizer
    try:
        logger.info(f"Loading model and tokenizer from: {MODEL_DIR}")
        if args.quantize:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        return
    
    # Evaluate the model on ANLI
    try:
        results = evaluate_gemma_on_anli(model, tokenizer, dataset, args)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return
    
    # Save results
    try:
        results.to_csv(args.output, index=False)
        logger.info(f"Results saved to '{args.output}'")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Perform analysis, save as JSON
    try:
        analysis = analyze_results(results)
        analysis_file = args.output.replace(".csv", "_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to '{analysis_file}'")
        
        # Print summary statistics
        print("\nEvaluation Summary:")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Accuracy: {analysis['accuracy']*100:.1f}% (Baseline: random {analysis['random_baseline']*100:.1f}%, majority {analysis['majority_baseline']*100:.1f}%)")
        print("\nClass metrics:")
        for label, metrics in analysis['class_metrics'].items():
            print(f"  {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1']:.2f}, support={metrics['support']}")
        print("\nRound metrics:")
        for round_name, metrics in analysis['round_metrics'].items():
            print(f"  {round_name}: accuracy={metrics['accuracy']*100:.1f}%, examples={metrics['examples']}")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()