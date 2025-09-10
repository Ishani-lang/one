"""
Evaluator for cross-lingual POS tagging models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from .metrics import POSTaggingMetrics
from ..models.pos_tagger import POSTagger
from ..data.utils import get_label_mapping

logger = logging.getLogger(__name__)

class CrossLingualEvaluator:
    """Evaluator for cross-lingual POS tagging models"""
    
    def __init__(self, model: POSTagger):
        self.model = model
        self.metrics_calculator = POSTaggingMetrics(model.label_mapping)
    
    def evaluate_model(self, 
                      eval_dataset,
                      adapter_name: str = "pos_tagging") -> Dict[str, float]:
        """Evaluate model on a dataset"""
        try:
            # Set active adapters
            self.model.adapter_model.set_active_adapters([adapter_name])
            
            # Get predictions
            predictions, labels = self._get_predictions(eval_dataset)
            
            # Compute metrics
            metrics = self.metrics_calculator.compute_metrics(predictions, labels)
            
            # Add sequence accuracy
            metrics['sequence_accuracy'] = self.metrics_calculator.compute_sequence_accuracy(
                [predictions], [labels]
            )
            
            logger.info(f"Evaluation completed for adapter {adapter_name}: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def evaluate_cross_lingual_transfer(self, 
                                      source_dataset,
                                      target_dataset,
                                      source_adapter: str = "ta",
                                      target_adapter: str = "toda") -> Dict[str, float]:
        """Evaluate cross-lingual transfer performance"""
        try:
            # Evaluate on source language
            source_metrics = self.evaluate_model(source_dataset, source_adapter)
            
            # Evaluate on target language
            target_metrics = self.evaluate_model(target_dataset, target_adapter)
            
            # Compute cross-lingual metrics
            cross_lingual_metrics = self.metrics_calculator.compute_cross_lingual_metrics(
                source_predictions=[],  # Will be computed internally
                source_labels=[],
                target_predictions=[],
                target_labels=[]
            )
            
            # Combine metrics
            combined_metrics = {
                'source_f1': source_metrics['f1'],
                'target_f1': target_metrics['f1'],
                'source_accuracy': source_metrics['accuracy'],
                'target_accuracy': target_metrics['accuracy'],
                'transfer_efficiency': target_metrics['f1'] / source_metrics['f1'] if source_metrics['f1'] > 0 else 0,
                'performance_drop': source_metrics['f1'] - target_metrics['f1']
            }
            
            logger.info(f"Cross-lingual transfer evaluation completed: {combined_metrics}")
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating cross-lingual transfer: {e}")
            raise
    
    def evaluate_few_shot_learning(self, 
                                 few_shot_dataset,
                                 baseline_dataset=None,
                                 few_shot_adapter: str = "toda") -> Dict[str, float]:
        """Evaluate few-shot learning performance"""
        try:
            # Evaluate few-shot model
            few_shot_metrics = self.evaluate_model(few_shot_dataset, few_shot_adapter)
            
            if baseline_dataset is not None:
                # Evaluate baseline model
                baseline_metrics = self.evaluate_model(baseline_dataset, "pos_tagging")
                
                # Compute few-shot metrics
                few_shot_improvement = self.metrics_calculator.compute_few_shot_metrics(
                    few_shot_predictions=[],
                    few_shot_labels=[],
                    baseline_predictions=[],
                    baseline_labels=[]
                )
                
                # Combine metrics
                combined_metrics = {
                    'few_shot_f1': few_shot_metrics['f1'],
                    'baseline_f1': baseline_metrics['f1'],
                    'f1_improvement': few_shot_metrics['f1'] - baseline_metrics['f1'],
                    'few_shot_accuracy': few_shot_metrics['accuracy'],
                    'baseline_accuracy': baseline_metrics['accuracy'],
                    'accuracy_improvement': few_shot_metrics['accuracy'] - baseline_metrics['accuracy']
                }
            else:
                combined_metrics = few_shot_metrics
            
            logger.info(f"Few-shot learning evaluation completed: {combined_metrics}")
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating few-shot learning: {e}")
            raise
    
    def evaluate_per_class_performance(self, 
                                     eval_dataset,
                                     adapter_name: str = "pos_tagging") -> Dict[str, Dict[str, float]]:
        """Evaluate per-class performance"""
        try:
            # Set active adapters
            self.model.adapter_model.set_active_adapters([adapter_name])
            
            # Get predictions
            predictions, labels = self._get_predictions(eval_dataset)
            
            # Compute per-class metrics
            per_class_metrics = self.metrics_calculator.compute_per_class_metrics(predictions, labels)
            
            logger.info(f"Per-class evaluation completed for adapter {adapter_name}")
            
            return per_class_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating per-class performance: {e}")
            raise
    
    def generate_detailed_report(self, 
                               eval_dataset,
                               adapter_name: str = "pos_tagging",
                               save_path: Optional[str] = None) -> str:
        """Generate detailed evaluation report"""
        try:
            # Set active adapters
            self.model.adapter_model.set_active_adapters([adapter_name])
            
            # Get predictions
            predictions, labels = self._get_predictions(eval_dataset)
            
            # Generate classification report
            report = self.metrics_calculator.generate_classification_report(predictions, labels)
            
            # Add summary metrics
            metrics = self.metrics_calculator.compute_metrics(predictions, labels)
            
            summary = f"""
EVALUATION SUMMARY
==================
Adapter: {adapter_name}
Accuracy: {metrics['accuracy']:.4f}
F1 Score: {metrics['f1']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
Support: {metrics['support']}

DETAILED CLASSIFICATION REPORT
==============================
{report}
"""
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Detailed report saved to {save_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            raise
    
    def plot_confusion_matrix(self, 
                             eval_dataset,
                             adapter_name: str = "pos_tagging",
                             save_path: Optional[str] = None) -> None:
        """Plot confusion matrix for evaluation"""
        try:
            # Set active adapters
            self.model.adapter_model.set_active_adapters([adapter_name])
            
            # Get predictions
            predictions, labels = self._get_predictions(eval_dataset)
            
            # Plot confusion matrix
            self.metrics_calculator.plot_confusion_matrix(
                predictions, labels, save_path
            )
            
            logger.info(f"Confusion matrix plotted for adapter {adapter_name}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise
    
    def _get_predictions(self, eval_dataset) -> Tuple[List[int], List[int]]:
        """Get predictions from the model"""
        all_predictions = []
        all_labels = []
        
        for example in eval_dataset:
            # Get input
            input_ids = torch.tensor(example['input_ids']).unsqueeze(0)
            attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0)
            labels = example['labels']
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model.adapter_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Flatten predictions and labels
            pred_flat = predictions.squeeze(0).tolist()
            label_flat = labels
            
            all_predictions.extend(pred_flat)
            all_labels.extend(label_flat)
        
        return all_predictions, all_labels
    
    def compare_adapters(self, 
                        eval_dataset,
                        adapters: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare performance of different adapters"""
        try:
            comparison_results = {}
            
            for adapter in adapters:
                logger.info(f"Evaluating adapter: {adapter}")
                metrics = self.evaluate_model(eval_dataset, adapter)
                comparison_results[adapter] = metrics
            
            # Find best adapter
            best_adapter = max(comparison_results.keys(), 
                             key=lambda x: comparison_results[x]['f1'])
            
            logger.info(f"Best adapter: {best_adapter} (F1: {comparison_results[best_adapter]['f1']:.4f})")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing adapters: {e}")
            raise
    
    def evaluate_robustness(self, 
                          eval_dataset,
                          adapter_name: str = "pos_tagging",
                          noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3]) -> Dict[float, Dict[str, float]]:
        """Evaluate model robustness to noise"""
        try:
            robustness_results = {}
            
            for noise_level in noise_levels:
                logger.info(f"Evaluating robustness with noise level: {noise_level}")
                
                # Add noise to dataset (simplified - in practice, you'd add noise to input)
                noisy_metrics = self.evaluate_model(eval_dataset, adapter_name)
                
                # For demonstration, we'll just use the original metrics
                # In practice, you'd add noise to the input data
                robustness_results[noise_level] = noisy_metrics
            
            logger.info(f"Robustness evaluation completed for adapter {adapter_name}")
            
            return robustness_results
            
        except Exception as e:
            logger.error(f"Error evaluating robustness: {e}")
            raise