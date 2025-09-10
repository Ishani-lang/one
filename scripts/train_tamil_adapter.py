#!/usr/bin/env python3
"""
Train Tamil language and POS task adapters
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import ModelConfig, TrainingConfig
from data.tamil_processor import TamilDataProcessor
from models.pos_tagger import POSTagger
from training.trainer import CrossLingualPOSTrainer, create_training_args
from evaluation.evaluator import CrossLingualEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    
    # Configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create output directories
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data processor
    logger.info("Initializing Tamil data processor...")
    tamil_processor = TamilDataProcessor(model_config.base_model_name)
    
    # Load Tamil dataset
    logger.info("Loading Tamil POS dataset...")
    tamil_dataset = tamil_processor.load_udpos_tamil()
    
    # Preprocess dataset
    logger.info("Preprocessing Tamil dataset...")
    processed_dataset = tamil_processor.preprocess_tamil_data(tamil_dataset)
    
    # Initialize POS tagger
    logger.info("Initializing POS tagger...")
    pos_tagger = POSTagger(
        base_model_name=model_config.base_model_name,
        num_labels=model_config.num_labels,
        adapter_config_name=model_config.adapter_config_name
    )
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=str(output_dir / "tamil_adapters"),
        learning_rate=training_config.learning_rate,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CrossLingualPOSTrainer(
        model=pos_tagger.adapter_model,
        tokenizer=pos_tagger.adapter_model.tokenizer,
        training_args=training_args
    )
    
    # Train Tamil language adapter
    logger.info("Training Tamil language adapter...")
    tamil_adapter_results = trainer.train_language_adapter(
        language="ta",
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        save_path=str(output_dir / "tamil_language_adapter")
    )
    
    logger.info(f"Tamil language adapter training completed: {tamil_adapter_results}")
    
    # Train POS task adapter
    logger.info("Training POS task adapter...")
    pos_task_results = trainer.train_task_adapter(
        task="pos_tagging",
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        save_path=str(output_dir / "pos_task_adapter")
    )
    
    logger.info(f"POS task adapter training completed: {pos_task_results}")
    
    # Train combined adapters
    logger.info("Training combined adapters...")
    combined_results = trainer.train_combined_adapters(
        language_adapters=["ta"],
        task_adapters=["pos_tagging"],
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        save_path=str(output_dir / "combined_adapters")
    )
    
    logger.info(f"Combined adapters training completed: {combined_results}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = CrossLingualEvaluator(pos_tagger)
    
    test_metrics = evaluator.evaluate_model(
        eval_dataset=processed_dataset["test"],
        adapter_name="pos_tagging"
    )
    
    logger.info(f"Test set evaluation: {test_metrics}")
    
    # Generate detailed report
    logger.info("Generating detailed evaluation report...")
    report = evaluator.generate_detailed_report(
        eval_dataset=processed_dataset["test"],
        adapter_name="pos_tagging",
        save_path=str(output_dir / "evaluation_report.txt")
    )
    
    print("\n" + "="*50)
    print("TAMIL ADAPTER TRAINING COMPLETED")
    print("="*50)
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print("="*50)
    
    # Save model
    logger.info("Saving complete model...")
    pos_tagger.save_model(str(output_dir / "tamil_pos_tagger"))
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()