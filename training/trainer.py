"""
Training utilities for cross-lingual POS tagging
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.adapters import AdapterTrainer
from typing import Dict, List, Optional, Union
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

logger = logging.getLogger(__name__)

class CrossLingualPOSTrainer:
    """Custom trainer for cross-lingual POS tagging with adapters"""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 training_args: TrainingArguments,
                 compute_metrics_fn=None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.compute_metrics_fn = compute_metrics_fn or self._default_compute_metrics
        
        # Initialize trainer
        self.trainer = AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def _default_compute_metrics(self, eval_pred):
        """Default metrics computation for POS tagging"""
        predictions, labels = eval_pred
        
        # Remove ignored labels (-100)
        predictions = np.argmax(predictions, axis=2)
        
        # Flatten predictions and labels
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_predictions.append(pred)
                    true_labels.append(lab)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, true_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_adapter(self, 
                     adapter_name: str,
                     train_dataset,
                     eval_dataset=None,
                     save_path: Optional[str] = None) -> Dict[str, float]:
        """Train a specific adapter"""
        try:
            # Set active adapters
            self.model.set_active_adapters([adapter_name])
            
            # Update trainer datasets
            self.trainer.train_dataset = train_dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Train the adapter
            logger.info(f"Training adapter: {adapter_name}")
            train_results = self.trainer.train()
            
            # Save adapter if path provided
            if save_path:
                self.model.save_adapter(save_path, adapter_name)
                logger.info(f"Saved adapter {adapter_name} to {save_path}")
            
            return train_results.metrics
            
        except Exception as e:
            logger.error(f"Error training adapter {adapter_name}: {e}")
            raise
    
    def evaluate_adapter(self, 
                        adapter_name: str,
                        eval_dataset) -> Dict[str, float]:
        """Evaluate a specific adapter"""
        try:
            # Set active adapters
            self.model.set_active_adapters([adapter_name])
            
            # Update trainer eval dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Evaluate the adapter
            logger.info(f"Evaluating adapter: {adapter_name}")
            eval_results = self.trainer.evaluate()
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error evaluating adapter {adapter_name}: {e}")
            raise
    
    def train_language_adapter(self, 
                              language: str,
                              train_dataset,
                              eval_dataset=None,
                              save_path: Optional[str] = None) -> Dict[str, float]:
        """Train language-specific adapter"""
        return self.train_adapter(
            adapter_name=language,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            save_path=save_path
        )
    
    def train_task_adapter(self, 
                          task: str,
                          train_dataset,
                          eval_dataset=None,
                          save_path: Optional[str] = None) -> Dict[str, float]:
        """Train task-specific adapter"""
        return self.train_adapter(
            adapter_name=task,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            save_path=save_path
        )
    
    def train_combined_adapters(self, 
                               language_adapters: List[str],
                               task_adapters: List[str],
                               train_dataset,
                               eval_dataset=None,
                               save_path: Optional[str] = None) -> Dict[str, float]:
        """Train combined language and task adapters"""
        try:
            # Set active adapters
            active_adapters = language_adapters + task_adapters
            self.model.set_active_adapters(active_adapters)
            
            # Update trainer datasets
            self.trainer.train_dataset = train_dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Train the combined adapters
            logger.info(f"Training combined adapters: {active_adapters}")
            train_results = self.trainer.train()
            
            # Save adapters if path provided
            if save_path:
                for adapter in active_adapters:
                    adapter_save_path = os.path.join(save_path, adapter)
                    self.model.save_adapter(adapter_save_path, adapter)
                logger.info(f"Saved adapters to {save_path}")
            
            return train_results.metrics
            
        except Exception as e:
            logger.error(f"Error training combined adapters: {e}")
            raise

class FewShotTrainer:
    """Trainer for few-shot learning scenarios"""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 training_args: TrainingArguments):
        
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        
        # Initialize trainer
        self.trainer = AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    
    def train_few_shot(self, 
                      adapter_name: str,
                      few_shot_dataset,
                      base_adapters: List[str] = None) -> Dict[str, float]:
        """Train adapter with few-shot learning"""
        try:
            # Set base adapters if provided
            if base_adapters:
                self.model.set_active_adapters(base_adapters)
            
            # Add new adapter for few-shot learning
            if adapter_name not in self.model.active_adapters:
                self.model.add_adapter(adapter_name)
                self.model.train_adapter([adapter_name])
            
            # Set active adapters
            active_adapters = (base_adapters or []) + [adapter_name]
            self.model.set_active_adapters(active_adapters)
            
            # Update trainer dataset
            self.trainer.train_dataset = few_shot_dataset
            
            # Train with few-shot data
            logger.info(f"Training few-shot adapter: {adapter_name}")
            train_results = self.trainer.train()
            
            return train_results.metrics
            
        except Exception as e:
            logger.error(f"Error training few-shot adapter {adapter_name}: {e}")
            raise
    
    def evaluate_few_shot(self, 
                         adapter_name: str,
                         eval_dataset,
                         base_adapters: List[str] = None) -> Dict[str, float]:
        """Evaluate few-shot adapter"""
        try:
            # Set active adapters
            active_adapters = (base_adapters or []) + [adapter_name]
            self.model.set_active_adapters(active_adapters)
            
            # Update trainer eval dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Evaluate
            logger.info(f"Evaluating few-shot adapter: {adapter_name}")
            eval_results = self.trainer.evaluate()
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error evaluating few-shot adapter {adapter_name}: {e}")
            raise

def create_training_args(output_dir: str,
                        learning_rate: float = 2e-5,
                        num_train_epochs: int = 3,
                        per_device_train_batch_size: int = 16,
                        per_device_eval_batch_size: int = 16,
                        warmup_steps: int = 500,
                        weight_decay: float = 0.01,
                        evaluation_strategy: str = "epoch",
                        save_strategy: str = "epoch",
                        save_total_limit: int = 3,
                        logging_dir: str = None,
                        **kwargs) -> TrainingArguments:
    """Create training arguments with sensible defaults"""
    
    if logging_dir is None:
        logging_dir = os.path.join(output_dir, "logs")
    
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_dir=logging_dir,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        **kwargs
    )

def create_few_shot_training_args(output_dir: str,
                                 learning_rate: float = 1e-5,
                                 num_train_epochs: int = 10,
                                 per_device_train_batch_size: int = 8,
                                 per_device_eval_batch_size: int = 8,
                                 warmup_steps: int = 100,
                                 weight_decay: float = 0.01,
                                 **kwargs) -> TrainingArguments:
    """Create training arguments optimized for few-shot learning"""
    
    return create_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        evaluation_strategy="no",  # No evaluation during few-shot training
        save_strategy="epoch",
        save_total_limit=1,
        **kwargs
    )