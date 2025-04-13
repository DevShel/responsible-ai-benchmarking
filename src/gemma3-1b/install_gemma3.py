import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Install Gemma3 1b model locally")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it",
                        help="Hugging Face model identifier")
    parser.add_argument("--save_dir", type=str, default="gemma3_model",
                        help="Directory where the model and tokenizer will be saved")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for loading the model")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization to reduce memory usage")
    return parser.parse_args()

def install_model(args):
    logger.info("Installing Gemma3 1b model...")
    
    try:
        # Load the model (with optional quantization)
        if args.quantize:
            logger.info("Using 4-bit quantization")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Save the model and tokenizer locally
        logger.info("Saving model and tokenizer locally...")
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        logger.info(f"Model installed and saved in '{args.save_dir}'")
    except Exception as e:
        logger.error(f"Error during installation: {e}")

def main():
    args = parse_args()
    install_model(args)

if __name__ == "__main__":
    main()