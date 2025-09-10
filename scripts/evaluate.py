#!/usr/bin/env python3
"""
Evaluate the trained POS tagger models
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import ModelConfig
from data.tamil_processor import TamilDataProcessor
from data.toda_processor import TodaDataProcessor
from models.pos_tagger import POSTagger
from evaluation.evaluator import CrossLingualEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_tamil_model(model_path: str, output_dir: str):
    """Evaluate Tamil POS tagger model"""
    
    logger.info(f"Evaluating Tamil model from {model_path}")
    
    # Initialize model
    pos_tagger = POSTagger()
    pos_tagger.load_model(model_path)
    
    # Initialize data processor
    tamil_processor = TamilDataProcessor()
    
    # Load Tamil dataset
    tamil_dataset = tamil_processor.load_udpos_tamil()
    processed_dataset = tamil_processor.preprocess_tamil_data(tamil_dataset)
    
    # Initialize evaluator
    evaluator = CrossLingualEvaluator(pos_tagger)
    
    # Evaluate on different splits
    splits = ["train", "validation", "test"]
    results = {}
    
    for split in splits:
        logger.info(f"Evaluating on {split} set...")
        metrics = evaluator.evaluate_model(
            eval_dataset=processed_dataset[split],
            adapter_name="pos_tagging"
        )
        results[split] = metrics
        
        print(f"\n{split.upper()} SET RESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(
        eval_dataset=processed_dataset["test"],
        adapter_name="pos_tagging",
        save_path=os.path.join(output_dir, "tamil_evaluation_report.txt")
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        eval_dataset=processed_dataset["test"],
        adapter_name="pos_tagging",
        save_path=os.path.join(output_dir, "tamil_confusion_matrix.png")
    )
    
    return results

def evaluate_toda_model(model_path: str, output_dir: str):
    """Evaluate Toda POS tagger model"""
    
    logger.info(f"Evaluating Toda model from {model_path}")
    
    # Initialize model
    pos_tagger = POSTagger()
    pos_tagger.load_model(model_path)
    
    # Initialize data processor
    toda_processor = TodaDataProcessor()
    
    # Create Toda dataset
    toda_dataset = toda_processor.create_toda_few_shot_dataset(num_samples=100)
    processed_dataset = toda_processor.preprocess_toda_data(toda_dataset)
    
    # Initialize evaluator
    evaluator = CrossLingualEvaluator(pos_tagger)
    
    # Evaluate different adapters
    adapters = ["toda", "toda_transfer", "pos_tagging"]
    results = {}
    
    for adapter in adapters:
        try:
            logger.info(f"Evaluating adapter: {adapter}")
            metrics = evaluator.evaluate_model(
                eval_dataset=processed_dataset["test"],
                adapter_name=adapter
            )
            results[adapter] = metrics
            
            print(f"\n{adapter.upper()} ADAPTER RESULTS:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
        except Exception as e:
            logger.warning(f"Could not evaluate adapter {adapter}: {e}")
            results[adapter] = None
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(
        eval_dataset=processed_dataset["test"],
        adapter_name="toda",
        save_path=os.path.join(output_dir, "toda_evaluation_report.txt")
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        eval_dataset=processed_dataset["test"],
        adapter_name="toda",
        save_path=os.path.join(output_dir, "toda_confusion_matrix.png")
    )
    
    return results

def compare_models(tamil_model_path: str, toda_model_path: str, output_dir: str):
    """Compare Tamil and Toda models"""
    
    logger.info("Comparing Tamil and Toda models")
    
    # Load models
    tamil_tagger = POSTagger()
    tamil_tagger.load_model(tamil_model_path)
    
    toda_tagger = POSTagger()
    toda_tagger.load_model(toda_model_path)
    
    # Create test dataset
    toda_processor = TodaDataProcessor()
    toda_dataset = toda_processor.create_toda_few_shot_dataset(num_samples=50)
    processed_dataset = toda_processor.preprocess_toda_data(toda_dataset)
    
    # Initialize evaluators
    tamil_evaluator = CrossLingualEvaluator(tamil_tagger)
    toda_evaluator = CrossLingualEvaluator(toda_tagger)
    
    # Evaluate both models
    tamil_metrics = tamil_evaluator.evaluate_model(
        eval_dataset=processed_dataset["test"],
        adapter_name="pos_tagging"
    )
    
    toda_metrics = toda_evaluator.evaluate_model(
        eval_dataset=processed_dataset["test"],
        adapter_name="toda"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON ON TODA TEST SET")
    print("="*60)
    print(f"{'Metric':<15} {'Tamil Model':<15} {'Toda Model':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        tamil_val = tamil_metrics[metric]
        toda_val = toda_metrics[metric]
        improvement = toda_val - tamil_val
        print(f"{metric.capitalize():<15} {tamil_val:<15.4f} {toda_val:<15.4f} {improvement:<15.4f}")
    
    print("="*60)
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, "model_comparison.txt")
    with open(comparison_file, 'w') as f:
        f.write("MODEL COMPARISON ON TODA TEST SET\n")
        f.write("="*60 + "\n")
        f.write(f"{'Metric':<15} {'Tamil Model':<15} {'Toda Model':<15} {'Improvement':<15}\n")
        f.write("-" * 60 + "\n")
        
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            tamil_val = tamil_metrics[metric]
            toda_val = toda_metrics[metric]
            improvement = toda_val - tamil_val
            f.write(f"{metric.capitalize():<15} {tamil_val:<15.4f} {toda_val:<15.4f} {improvement:<15.4f}\n")
    
    logger.info(f"Comparison results saved to {comparison_file}")

def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description="Evaluate POS tagger models")
    parser.add_argument("--tamil-model", type=str, help="Path to Tamil model")
    parser.add_argument("--toda-model", type=str, help="Path to Toda model")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", 
                       help="Output directory for evaluation results")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare Tamil and Toda models")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate Tamil model
    if args.tamil_model:
        logger.info("Evaluating Tamil model...")
        tamil_results = evaluate_tamil_model(args.tamil_model, str(output_dir))
    
    # Evaluate Toda model
    if args.toda_model:
        logger.info("Evaluating Toda model...")
        toda_results = evaluate_toda_model(args.toda_model, str(output_dir))
    
    # Compare models
    if args.compare and args.tamil_model and args.toda_model:
        logger.info("Comparing models...")
        compare_models(args.tamil_model, args.toda_model, str(output_dir))
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()