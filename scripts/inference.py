#!/usr/bin/env python3
"""
Inference script for Toda POS tagger
"""

import os
import sys
import logging
from pathlib import Path
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.pos_tagger import POSTagger
from config.model_config import UNIVERSAL_POS_TAGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> POSTagger:
    """Load the trained POS tagger model"""
    logger.info(f"Loading model from {model_path}")
    
    pos_tagger = POSTagger()
    pos_tagger.load_model(model_path)
    
    logger.info("Model loaded successfully")
    return pos_tagger

def predict_single_sentence(model: POSTagger, 
                           sentence: str, 
                           adapter_name: str = "toda") -> list:
    """Predict POS tags for a single sentence"""
    
    # Set active adapters
    model.adapter_model.set_active_adapters([adapter_name])
    
    # Predict POS tags
    predictions = model.predict_pos_tags(sentence)
    
    return predictions

def predict_batch(model: POSTagger, 
                 sentences: list, 
                 adapter_name: str = "toda") -> list:
    """Predict POS tags for a batch of sentences"""
    
    # Set active adapters
    model.adapter_model.set_active_adapters([adapter_name])
    
    # Predict POS tags for all sentences
    all_predictions = []
    for sentence in sentences:
        predictions = model.predict_pos_tags(sentence)
        all_predictions.append(predictions)
    
    return all_predictions

def interactive_mode(model: POSTagger, adapter_name: str = "toda"):
    """Interactive mode for POS tagging"""
    
    print("\n" + "="*50)
    print("TODA POS TAGGER - INTERACTIVE MODE")
    print("="*50)
    print("Enter sentences to tag (type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            # Get user input
            sentence = input("\nEnter sentence: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not sentence:
                continue
            
            # Predict POS tags
            predictions = predict_single_sentence(model, sentence, adapter_name)
            
            # Display results
            print("\nPOS Tagging Results:")
            print("-" * 30)
            for token, tag in predictions:
                print(f"{token:<20} {tag}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def batch_mode(model: POSTagger, 
              input_file: str, 
              output_file: str, 
              adapter_name: str = "toda"):
    """Batch mode for POS tagging"""
    
    logger.info(f"Processing batch from {input_file}")
    
    # Read input sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Predict POS tags
    predictions = predict_batch(model, sentences, adapter_name)
    
    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (sentence, preds) in enumerate(zip(sentences, predictions)):
            f.write(f"Sentence {i+1}: {sentence}\n")
            f.write("POS Tags:\n")
            for token, tag in preds:
                f.write(f"  {token:<20} {tag}\n")
            f.write("\n")
    
    logger.info(f"Results saved to {output_file}")

def demo_mode(model: POSTagger, adapter_name: str = "toda"):
    """Demo mode with sample sentences"""
    
    print("\n" + "="*50)
    print("TODA POS TAGGER - DEMO MODE")
    print("="*50)
    
    # Sample Toda sentences (transliterated)
    sample_sentences = [
        "nān ōr pōḍa kōḍa āg irukān",
        "nī ēṇḍa pōḍa kōḍa āg irukāy",
        "nān pōḍa kōḍa āg irukān ēṇḍa",
        "nī pōḍa kōḍa āg irukāy ēṇḍa",
        "nān pōḍa kōḍa āg irukān ēṇḍa"
    ]
    
    print("Sample Toda sentences (transliterated):")
    print("-" * 40)
    
    for i, sentence in enumerate(sample_sentences, 1):
        print(f"\n{i}. {sentence}")
        
        # Predict POS tags
        predictions = predict_single_sentence(model, sentence, adapter_name)
        
        # Display results
        print("   POS Tags:")
        for token, tag in predictions:
            print(f"     {token:<15} {tag}")

def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description="Toda POS Tagger Inference")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--adapter", type=str, default="toda",
                       choices=["toda", "toda_transfer", "pos_tagging"],
                       help="Adapter to use for inference")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "demo"],
                       help="Inference mode")
    parser.add_argument("--input-file", type=str,
                       help="Input file for batch mode")
    parser.add_argument("--output-file", type=str,
                       help="Output file for batch mode")
    parser.add_argument("--sentence", type=str,
                       help="Single sentence to tag")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model Info: {json.dumps(model_info, indent=2)}")
    
    if args.mode == "interactive":
        interactive_mode(model, args.adapter)
    
    elif args.mode == "batch":
        if not args.input_file or not args.output_file:
            print("Error: --input-file and --output-file required for batch mode")
            return
        batch_mode(model, args.input_file, args.output_file, args.adapter)
    
    elif args.mode == "demo":
        demo_mode(model, args.adapter)
    
    elif args.sentence:
        # Single sentence mode
        predictions = predict_single_sentence(model, args.sentence, args.adapter)
        print(f"\nSentence: {args.sentence}")
        print("POS Tags:")
        print("-" * 20)
        for token, tag in predictions:
            print(f"{token:<15} {tag}")

if __name__ == "__main__":
    main()