"""
Evaluation metrics for POS tagging
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class POSTaggingMetrics:
    """Metrics calculator for POS tagging evaluation"""
    
    def __init__(self, label_mapping: Dict[int, str]):
        self.label_mapping = label_mapping
        self.id2label = {v: k for k, v in label_mapping.items()}
    
    def compute_metrics(self, 
                       predictions: List[int], 
                       labels: List[int],
                       ignore_label: int = -100) -> Dict[str, float]:
        """Compute standard metrics for POS tagging"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # Calculate metrics
        accuracy = accuracy_score(filtered_labels, filtered_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_labels, filtered_predictions, average='weighted'
        )
        
        return {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'support': int(np.sum(support))
        }
    
    def compute_per_class_metrics(self, 
                                 predictions: List[int], 
                                 labels: List[int],
                                 ignore_label: int = -100) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics for POS tagging"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return {}
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_labels, filtered_predictions, average=None
        )
        
        per_class_metrics = {}
        for i, (tag, prec, rec, f1_score, supp) in enumerate(
            zip(self.label_mapping.values(), precision, recall, f1, support)
        ):
            per_class_metrics[tag] = {
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1_score),
                'support': int(supp)
            }
        
        return per_class_metrics
    
    def compute_confusion_matrix(self, 
                                predictions: List[int], 
                                labels: List[int],
                                ignore_label: int = -100) -> np.ndarray:
        """Compute confusion matrix for POS tagging"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return np.array([])
        
        # Calculate confusion matrix
        cm = confusion_matrix(filtered_labels, filtered_predictions)
        return cm
    
    def plot_confusion_matrix(self, 
                             predictions: List[int], 
                             labels: List[int],
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot confusion matrix for POS tagging"""
        
        cm = self.compute_confusion_matrix(predictions, labels)
        
        if cm.size == 0:
            print("No valid predictions to plot")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Get label names
        label_names = list(self.label_mapping.values())
        
        # Plot confusion matrix
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=label_names,
                   yticklabels=label_names)
        
        plt.title('Confusion Matrix for POS Tagging')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def compute_sequence_accuracy(self, 
                                 predictions: List[List[int]], 
                                 labels: List[List[int]],
                                 ignore_label: int = -100) -> float:
        """Compute sequence-level accuracy (exact match)"""
        
        correct_sequences = 0
        total_sequences = len(predictions)
        
        for pred_seq, label_seq in zip(predictions, labels):
            # Filter out ignored labels
            filtered_pred = [p for p, l in zip(pred_seq, label_seq) if l != ignore_label]
            filtered_label = [l for l in label_seq if l != ignore_label]
            
            if len(filtered_pred) == len(filtered_label) and all(p == l for p, l in zip(filtered_pred, filtered_label)):
                correct_sequences += 1
        
        return correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    def compute_token_accuracy(self, 
                              predictions: List[int], 
                              labels: List[int],
                              ignore_label: int = -100) -> float:
        """Compute token-level accuracy"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return 0.0
        
        return accuracy_score(filtered_labels, filtered_predictions)
    
    def compute_class_wise_f1(self, 
                             predictions: List[int], 
                             labels: List[int],
                             ignore_label: int = -100) -> Dict[str, float]:
        """Compute F1 score for each POS class"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return {}
        
        # Calculate per-class F1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_labels, filtered_predictions, average=None
        )
        
        class_f1 = {}
        for i, (tag, f1_score) in enumerate(zip(self.label_mapping.values(), f1)):
            class_f1[tag] = float(f1_score)
        
        return class_f1
    
    def generate_classification_report(self, 
                                     predictions: List[int], 
                                     labels: List[int],
                                     ignore_label: int = -100) -> str:
        """Generate detailed classification report"""
        
        # Filter out ignored labels
        filtered_predictions = []
        filtered_labels = []
        
        for pred, label in zip(predictions, labels):
            if label != ignore_label:
                filtered_predictions.append(pred)
                filtered_labels.append(label)
        
        if not filtered_predictions:
            return "No valid predictions to report"
        
        # Generate report
        target_names = list(self.label_mapping.values())
        report = classification_report(
            filtered_labels, 
            filtered_predictions, 
            target_names=target_names,
            zero_division=0
        )
        
        return report
    
    def compute_cross_lingual_metrics(self, 
                                    source_predictions: List[int],
                                    source_labels: List[int],
                                    target_predictions: List[int],
                                    target_labels: List[int],
                                    ignore_label: int = -100) -> Dict[str, float]:
        """Compute cross-lingual transfer metrics"""
        
        # Compute metrics for source language
        source_metrics = self.compute_metrics(source_predictions, source_labels, ignore_label)
        
        # Compute metrics for target language
        target_metrics = self.compute_metrics(target_predictions, target_labels, ignore_label)
        
        # Compute transfer efficiency
        transfer_efficiency = target_metrics['f1'] / source_metrics['f1'] if source_metrics['f1'] > 0 else 0
        
        # Compute performance drop
        performance_drop = source_metrics['f1'] - target_metrics['f1']
        
        return {
            'source_f1': source_metrics['f1'],
            'target_f1': target_metrics['f1'],
            'transfer_efficiency': transfer_efficiency,
            'performance_drop': performance_drop,
            'source_accuracy': source_metrics['accuracy'],
            'target_accuracy': target_metrics['accuracy']
        }
    
    def compute_few_shot_metrics(self, 
                                few_shot_predictions: List[int],
                                few_shot_labels: List[int],
                                baseline_predictions: List[int],
                                baseline_labels: List[int],
                                ignore_label: int = -100) -> Dict[str, float]:
        """Compute few-shot learning metrics"""
        
        # Compute metrics for few-shot model
        few_shot_metrics = self.compute_metrics(few_shot_predictions, few_shot_labels, ignore_label)
        
        # Compute metrics for baseline model
        baseline_metrics = self.compute_metrics(baseline_predictions, baseline_labels, ignore_label)
        
        # Compute few-shot improvement
        f1_improvement = few_shot_metrics['f1'] - baseline_metrics['f1']
        accuracy_improvement = few_shot_metrics['accuracy'] - baseline_metrics['accuracy']
        
        # Compute relative improvement
        relative_f1_improvement = f1_improvement / baseline_metrics['f1'] if baseline_metrics['f1'] > 0 else 0
        relative_accuracy_improvement = accuracy_improvement / baseline_metrics['accuracy'] if baseline_metrics['accuracy'] > 0 else 0
        
        return {
            'few_shot_f1': few_shot_metrics['f1'],
            'baseline_f1': baseline_metrics['f1'],
            'f1_improvement': f1_improvement,
            'relative_f1_improvement': relative_f1_improvement,
            'few_shot_accuracy': few_shot_metrics['accuracy'],
            'baseline_accuracy': baseline_metrics['accuracy'],
            'accuracy_improvement': accuracy_improvement,
            'relative_accuracy_improvement': relative_accuracy_improvement
        }