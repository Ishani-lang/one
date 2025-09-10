"""
Few-shot learning trainer for cross-lingual POS tagging
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.adapters import AdapterTrainer
from typing import Dict, List, Optional, Union, Tuple
import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

logger = logging.getLogger(__name__)

class FewShotLearningTrainer:
    """Specialized trainer for few-shot learning scenarios"""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 training_args: TrainingArguments,
                 base_adapters: List[str] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.base_adapters = base_adapters or []
        
        # Initialize trainer
        self.trainer = AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for few-shot evaluation"""
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
    
    def train_few_shot_adapter(self, 
                              adapter_name: str,
                              few_shot_dataset,
                              eval_dataset=None,
                              save_path: Optional[str] = None) -> Dict[str, float]:
        """Train adapter with few-shot learning"""
        try:
            # Set base adapters
            if self.base_adapters:
                self.model.set_active_adapters(self.base_adapters)
            
            # Add new adapter for few-shot learning
            if adapter_name not in [adapter for adapter in self.model.active_adapters]:
                self.model.add_adapter(adapter_name)
                self.model.train_adapter([adapter_name])
            
            # Set active adapters (base + new)
            active_adapters = self.base_adapters + [adapter_name]
            self.model.set_active_adapters(active_adapters)
            
            # Update trainer datasets
            self.trainer.train_dataset = few_shot_dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Train with few-shot data
            logger.info(f"Training few-shot adapter: {adapter_name}")
            logger.info(f"Active adapters: {active_adapters}")
            logger.info(f"Training samples: {len(few_shot_dataset)}")
            
            train_results = self.trainer.train()
            
            # Save adapter if path provided
            if save_path:
                self.model.save_adapter(save_path, adapter_name)
                logger.info(f"Saved adapter {adapter_name} to {save_path}")
            
            return train_results.metrics
            
        except Exception as e:
            logger.error(f"Error training few-shot adapter {adapter_name}: {e}")
            raise
    
    def evaluate_few_shot_adapter(self, 
                                 adapter_name: str,
                                 eval_dataset) -> Dict[str, float]:
        """Evaluate few-shot adapter"""
        try:
            # Set active adapters
            active_adapters = self.base_adapters + [adapter_name]
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
    
    def cross_lingual_transfer(self, 
                              source_language: str,
                              target_language: str,
                              target_dataset,
                              eval_dataset=None,
                              save_path: Optional[str] = None) -> Dict[str, float]:
        """Perform cross-lingual transfer from source to target language"""
        try:
            # Set base adapters (source language + task)
            base_adapters = [source_language, "pos_tagging"]
            self.model.set_active_adapters(base_adapters)
            
            # Add target language adapter
            target_adapter_name = f"{target_language}_transfer"
            if target_adapter_name not in [adapter for adapter in self.model.active_adapters]:
                self.model.add_adapter(target_adapter_name)
                self.model.train_adapter([target_adapter_name])
            
            # Set active adapters (source + task + target)
            active_adapters = base_adapters + [target_adapter_name]
            self.model.set_active_adapters(active_adapters)
            
            # Update trainer datasets
            self.trainer.train_dataset = target_dataset
            self.trainer.eval_dataset = eval_dataset
            
            # Train cross-lingual transfer
            logger.info(f"Training cross-lingual transfer: {source_language} -> {target_language}")
            logger.info(f"Active adapters: {active_adapters}")
            
            train_results = self.trainer.train()
            
            # Save adapter if path provided
            if save_path:
                self.model.save_adapter(save_path, target_adapter_name)
                logger.info(f"Saved cross-lingual adapter to {save_path}")
            
            return train_results.metrics
            
        except Exception as e:
            logger.error(f"Error in cross-lingual transfer: {e}")
            raise

class MetaLearningTrainer:
    """Meta-learning trainer for few-shot scenarios"""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 training_args: TrainingArguments,
                 support_adapters: List[str] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.support_adapters = support_adapters or []
        
        # Initialize trainer
        self.trainer = AdapterTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for meta-learning evaluation"""
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
    
    def meta_train(self, 
                  meta_dataset: Dict[str, List],
                  eval_dataset=None,
                  save_path: Optional[str] = None) -> Dict[str, float]:
        """Meta-train on multiple few-shot tasks"""
        try:
            # Set support adapters
            if self.support_adapters:
                self.model.set_active_adapters(self.support_adapters)
            
            # Meta-training loop
            meta_results = defaultdict(list)
            
            for task_name, task_data in meta_dataset.items():
                logger.info(f"Meta-training on task: {task_name}")
                
                # Add task-specific adapter
                if task_name not in [adapter for adapter in self.model.active_adapters]:
                    self.model.add_adapter(task_name)
                    self.model.train_adapter([task_name])
                
                # Set active adapters
                active_adapters = self.support_adapters + [task_name]
                self.model.set_active_adapters(active_adapters)
                
                # Update trainer dataset
                self.trainer.train_dataset = task_data
                self.trainer.eval_dataset = eval_dataset
                
                # Train on task
                task_results = self.trainer.train()
                meta_results[task_name] = task_results.metrics
                
                # Save task adapter
                if save_path:
                    task_save_path = os.path.join(save_path, task_name)
                    self.model.save_adapter(task_save_path, task_name)
            
            # Calculate average meta-results
            avg_results = {}
            for metric in ['train_loss', 'train_runtime', 'train_samples_per_second']:
                values = [results.get(metric, 0) for results in meta_results.values()]
                avg_results[f"meta_{metric}"] = np.mean(values) if values else 0
            
            logger.info(f"Meta-training completed. Average results: {avg_results}")
            
            return avg_results
            
        except Exception as e:
            logger.error(f"Error in meta-training: {e}")
            raise
    
    def meta_evaluate(self, 
                     meta_eval_dataset: Dict[str, List]) -> Dict[str, float]:
        """Meta-evaluate on multiple few-shot tasks"""
        try:
            meta_eval_results = defaultdict(list)
            
            for task_name, task_data in meta_eval_dataset.items():
                logger.info(f"Meta-evaluating on task: {task_name}")
                
                # Set active adapters
                active_adapters = self.support_adapters + [task_name]
                self.model.set_active_adapters(active_adapters)
                
                # Update trainer eval dataset
                self.trainer.eval_dataset = task_data
                
                # Evaluate on task
                task_eval_results = self.trainer.evaluate()
                meta_eval_results[task_name] = task_eval_results
            
            # Calculate average meta-eval results
            avg_eval_results = {}
            for metric in ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']:
                values = [results.get(metric, 0) for results in meta_eval_results.values()]
                avg_eval_results[f"meta_{metric}"] = np.mean(values) if values else 0
            
            logger.info(f"Meta-evaluation completed. Average results: {avg_eval_results}")
            
            return avg_eval_results
            
        except Exception as e:
            logger.error(f"Error in meta-evaluation: {e}")
            raise

def create_few_shot_training_args(output_dir: str,
                                 learning_rate: float = 1e-5,
                                 num_train_epochs: int = 10,
                                 per_device_train_batch_size: int = 4,
                                 per_device_eval_batch_size: int = 4,
                                 warmup_steps: int = 50,
                                 weight_decay: float = 0.01,
                                 **kwargs) -> TrainingArguments:
    """Create training arguments optimized for few-shot learning"""
    
    return TrainingArguments(
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
        logging_steps=10,
        load_best_model_at_end=False,
        **kwargs
    )

def create_meta_learning_training_args(output_dir: str,
                                      learning_rate: float = 5e-6,
                                      num_train_epochs: int = 5,
                                      per_device_train_batch_size: int = 2,
                                      per_device_eval_batch_size: int = 2,
                                      warmup_steps: int = 25,
                                      weight_decay: float = 0.01,
                                      **kwargs) -> TrainingArguments:
    """Create training arguments optimized for meta-learning"""
    
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        load_best_model_at_end=False,
        **kwargs
    )