#!/usr/bin/env python3
"""
Train Toda language adapter with few-shot learning
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_config import ModelConfig, TrainingConfig
from data.toda_processor import TodaDataProcessor
from models.pos_tagger import POSTagger
from training.few_shot_trainer import FewShotLearningTrainer, create_few_shot_training_args
from evaluation.evaluator import CrossLingualEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main few-shot training function"""
    
    # Configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create output directories
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data processor
    logger.info("Initializing Toda data processor...")
    toda_processor = TodaDataProcessor(model_config.base_model_name)
    
    # Create Toda few-shot dataset
    logger.info("Creating Toda few-shot dataset...")
    toda_dataset = toda_processor.create_toda_few_shot_dataset(
        num_samples=training_config.few_shot_samples
    )
    
    # Preprocess dataset
    logger.info("Preprocessing Toda dataset...")
    processed_dataset = toda_processor.preprocess_toda_data(toda_dataset)
    
    # Augment dataset for better few-shot learning
    logger.info("Augmenting Toda dataset...")
    augmented_dataset = toda_processor.augment_toda_data(processed_dataset, augmentation_factor=2)
    
    # Initialize POS tagger
    logger.info("Initializing POS tagger...")
    pos_tagger = POSTagger(
        base_model_name=model_config.base_model_name,
        num_labels=model_config.num_labels,
        adapter_config_name=model_config.adapter_config_name
    )
    
    # Load pre-trained Tamil adapters
    logger.info("Loading pre-trained Tamil adapters...")
    try:
        pos_tagger.load_model(str(output_dir / "tamil_pos_tagger"))
        logger.info("Successfully loaded Tamil adapters")
    except Exception as e:
        logger.warning(f"Could not load Tamil adapters: {e}")
        logger.info("Proceeding with base model only")
    
    # Create few-shot training arguments
    training_args = create_few_shot_training_args(
        output_dir=str(output_dir / "toda_few_shot"),
        learning_rate=training_config.few_shot_learning_rate,
        num_train_epochs=training_config.few_shot_epochs,
        per_device_train_batch_size=4,  # Smaller batch size for few-shot
        per_device_eval_batch_size=4
    )
    
    # Initialize few-shot trainer
    logger.info("Initializing few-shot trainer...")
    few_shot_trainer = FewShotLearningTrainer(
        model=pos_tagger.adapter_model,
        tokenizer=pos_tagger.adapter_model.tokenizer,
        training_args=training_args,
        base_adapters=["ta", "pos_tagging"]  # Use Tamil adapters as base
    )
    
    # Train Toda language adapter with few-shot learning
    logger.info("Training Toda language adapter with few-shot learning...")
    toda_adapter_results = few_shot_trainer.train_few_shot_adapter(
        adapter_name="toda",
        few_shot_dataset=augmented_dataset["train"],
        eval_dataset=augmented_dataset["validation"],
        save_path=str(output_dir / "toda_language_adapter")
    )
    
    logger.info(f"Toda language adapter training completed: {toda_adapter_results}")
    
    # Perform cross-lingual transfer
    logger.info("Performing cross-lingual transfer...")
    transfer_results = few_shot_trainer.cross_lingual_transfer(
        source_language="ta",
        target_language="toda",
        target_dataset=augmented_dataset["train"],
        eval_dataset=augmented_dataset["validation"],
        save_path=str(output_dir / "cross_lingual_transfer")
    )
    
    logger.info(f"Cross-lingual transfer completed: {transfer_results}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = CrossLingualEvaluator(pos_tagger)
    
    # Evaluate Toda adapter
    toda_test_metrics = evaluator.evaluate_model(
        eval_dataset=augmented_dataset["test"],
        adapter_name="toda"
    )
    
    logger.info(f"Toda test set evaluation: {toda_test_metrics}")
    
    # Evaluate cross-lingual transfer
    transfer_test_metrics = evaluator.evaluate_model(
        eval_dataset=augmented_dataset["test"],
        adapter_name="toda_transfer"
    )
    
    logger.info(f"Cross-lingual transfer test evaluation: {transfer_test_metrics}")
    
    # Compare with baseline (Tamil adapters)
    baseline_metrics = evaluator.evaluate_model(
        eval_dataset=augmented_dataset["test"],
        adapter_name="pos_tagging"
    )
    
    logger.info(f"Baseline (Tamil) test evaluation: {baseline_metrics}")
    
    # Generate detailed report
    logger.info("Generating detailed evaluation report...")
    report = evaluator.generate_detailed_report(
        eval_dataset=augmented_dataset["test"],
        adapter_name="toda",
        save_path=str(output_dir / "toda_evaluation_report.txt")
    )
    
    # Plot confusion matrix
    logger.info("Generating confusion matrix...")
    evaluator.plot_confusion_matrix(
        eval_dataset=augmented_dataset["test"],
        adapter_name="toda",
        save_path=str(output_dir / "toda_confusion_matrix.png")
    )
    
    print("\n" + "="*50)
    print("TODA FEW-SHOT TRAINING COMPLETED")
    print("="*50)
    print(f"Toda F1 Score: {toda_test_metrics['f1']:.4f}")
    print(f"Toda Accuracy: {toda_test_metrics['accuracy']:.4f}")
    print(f"Transfer F1 Score: {transfer_test_metrics['f1']:.4f}")
    print(f"Transfer Accuracy: {transfer_test_metrics['accuracy']:.4f}")
    print(f"Baseline F1 Score: {baseline_metrics['f1']:.4f}")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print("="*50)
    
    # Save complete model
    logger.info("Saving complete model...")
    pos_tagger.save_model(str(output_dir / "toda_pos_tagger"))
    
    logger.info("Few-shot training completed successfully!")

if __name__ == "__main__":
    main()