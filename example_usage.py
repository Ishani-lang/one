#!/usr/bin/env python3
"""
Example usage of the Toda POS Tagger
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.pos_tagger import POSTagger
from data.tamil_processor import TamilDataProcessor
from data.toda_processor import TodaDataProcessor
from evaluation.evaluator import CrossLingualEvaluator

def main():
    """Example usage of the POS tagger"""
    
    print("="*60)
    print("TODA POS TAGGER - EXAMPLE USAGE")
    print("="*60)
    
    # 1. Initialize the POS tagger
    print("\n1. Initializing POS Tagger...")
    pos_tagger = POSTagger(
        base_model_name="xlm-roberta-base",
        num_labels=17,
        adapter_config_name="pfeiffer"
    )
    
    # 2. Load Tamil dataset
    print("\n2. Loading Tamil dataset...")
    tamil_processor = TamilDataProcessor()
    tamil_dataset = tamil_processor.load_udpos_tamil()
    processed_tamil = tamil_processor.preprocess_tamil_data(tamil_dataset)
    
    print(f"   Tamil dataset loaded: {len(processed_tamil['train'])} train samples")
    
    # 3. Load Toda dataset
    print("\n3. Creating Toda few-shot dataset...")
    toda_processor = TodaDataProcessor()
    toda_dataset = toda_processor.create_toda_few_shot_dataset(num_samples=50)
    processed_toda = toda_processor.preprocess_toda_data(toda_dataset)
    
    print(f"   Toda dataset created: {len(processed_toda['train'])} train samples")
    
    # 4. Setup Tamil adapters
    print("\n4. Setting up Tamil adapters...")
    pos_tagger.setup_tamil_adapters()
    
    # 5. Predict POS tags for sample sentences
    print("\n5. Predicting POS tags for sample sentences...")
    
    # Sample Tamil sentence
    tamil_sentence = "நான் ஒரு மாணவன்"
    tamil_predictions = pos_tagger.predict_pos_tags(tamil_sentence)
    
    print(f"\nTamil sentence: {tamil_sentence}")
    print("POS tags:")
    for token, tag in tamil_predictions:
        print(f"  {token:<15} {tag}")
    
    # Sample Toda sentence (transliterated)
    toda_sentence = "nān ōr pōḍa kōḍa"
    toda_predictions = pos_tagger.predict_pos_tags(toda_sentence)
    
    print(f"\nToda sentence: {toda_sentence}")
    print("POS tags:")
    for token, tag in toda_predictions:
        print(f"  {token:<15} {tag}")
    
    # 6. Evaluate model performance
    print("\n6. Evaluating model performance...")
    evaluator = CrossLingualEvaluator(pos_tagger)
    
    # Evaluate on Tamil test set
    tamil_metrics = evaluator.evaluate_model(
        eval_dataset=processed_tamil["test"],
        adapter_name="pos_tagging"
    )
    
    print(f"\nTamil test set performance:")
    print(f"  Accuracy: {tamil_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {tamil_metrics['f1']:.4f}")
    print(f"  Precision: {tamil_metrics['precision']:.4f}")
    print(f"  Recall: {tamil_metrics['recall']:.4f}")
    
    # 7. Model information
    print("\n7. Model information:")
    model_info = pos_tagger.get_model_info()
    print(f"  Base model: {model_info['base_model']}")
    print(f"  Number of labels: {model_info['num_labels']}")
    print(f"  Adapter config: {model_info['adapter_config']}")
    print(f"  Available adapters: {model_info['adapters']}")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Train Tamil adapters: python scripts/train_tamil_adapter.py")
    print("2. Train Toda few-shot: python scripts/train_toda_fewshot.py")
    print("3. Evaluate models: python scripts/evaluate.py")
    print("4. Run inference: python scripts/inference.py")

if __name__ == "__main__":
    main()