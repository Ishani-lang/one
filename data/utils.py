"""
Utility functions for data processing
"""

import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def align_labels_with_tokens(labels: List[str], word_ids: List[Optional[int]]) -> List[int]:
    """
    Align labels with tokenized input, handling subword tokenization
    
    Args:
        labels: List of original labels
        word_ids: List of word indices from tokenizer
        
    Returns:
        List of aligned label IDs
    """
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens (CLS, SEP, PAD) get -100 label
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # New word, use the label
            aligned_labels.append(labels[word_idx])
        else:
            # Subword token, use -100 to ignore in loss calculation
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    
    return aligned_labels

def compute_metrics(eval_pred):
    """
    Compute metrics for POS tagging evaluation
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
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

def get_label_mapping() -> Dict[int, str]:
    """Get mapping from label IDs to POS tag names"""
    return {
        0: "ADJ",    # adjective
        1: "ADP",    # adposition
        2: "ADV",    # adverb
        3: "AUX",    # auxiliary
        4: "CCONJ",  # coordinating conjunction
        5: "DET",    # determiner
        6: "INTJ",   # interjection
        7: "NOUN",   # noun
        8: "NUM",    # numeral
        9: "PART",   # particle
        10: "PRON",  # pronoun
        11: "PROPN", # proper noun
        12: "PUNCT", # punctuation
        13: "SCONJ", # subordinating conjunction
        14: "SYM",   # symbol
        15: "VERB",  # verb
        16: "X"      # other
    }

def format_predictions(predictions: List[int], tokens: List[str], label_mapping: Dict[int, str]) -> List[Tuple[str, str]]:
    """
    Format predictions for display
    
    Args:
        predictions: List of predicted label IDs
        tokens: List of tokens
        label_mapping: Mapping from label IDs to tag names
        
    Returns:
        List of (token, tag) tuples
    """
    formatted = []
    for token, pred in zip(tokens, predictions):
        if pred != -100:  # Skip ignored labels
            tag = label_mapping.get(pred, "UNK")
            formatted.append((token, tag))
    return formatted

def create_confusion_matrix(labels: List[int], predictions: List[int], label_mapping: Dict[int, str]) -> np.ndarray:
    """
    Create confusion matrix for POS tagging evaluation
    
    Args:
        labels: True labels
        predictions: Predicted labels
        label_mapping: Mapping from label IDs to tag names
        
    Returns:
        Confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    # Filter out ignored labels
    filtered_labels = []
    filtered_predictions = []
    
    for label, pred in zip(labels, predictions):
        if label != -100:
            filtered_labels.append(label)
            filtered_predictions.append(pred)
    
    return confusion_matrix(filtered_labels, filtered_predictions)

def calculate_per_class_metrics(labels: List[int], predictions: List[int], label_mapping: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics for POS tagging
    
    Args:
        labels: True labels
        predictions: Predicted labels
        label_mapping: Mapping from label IDs to tag names
        
    Returns:
        Dictionary of per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Filter out ignored labels
    filtered_labels = []
    filtered_predictions = []
    
    for label, pred in zip(labels, predictions):
        if label != -100:
            filtered_labels.append(label)
            filtered_predictions.append(pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_labels, filtered_predictions, average=None
    )
    
    per_class_metrics = {}
    for i, (tag, prec, rec, f1_score, supp) in enumerate(zip(label_mapping.values(), precision, recall, f1, support)):
        per_class_metrics[tag] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1_score),
            'support': int(supp)
        }
    
    return per_class_metrics

def save_predictions(predictions: List[Tuple[str, str]], filename: str):
    """
    Save predictions to file
    
    Args:
        predictions: List of (token, tag) tuples
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for token, tag in predictions:
            f.write(f"{token}\t{tag}\n")

def load_predictions(filename: str) -> List[Tuple[str, str]]:
    """
    Load predictions from file
    
    Args:
        filename: Input filename
        
    Returns:
        List of (token, tag) tuples
    """
    predictions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                token, tag = line.split('\t')
                predictions.append((token, tag))
    return predictions