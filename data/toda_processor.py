"""
Toda language data processing for few-shot learning
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re
from .utils import align_labels_with_tokens

class TodaDataProcessor:
    """Data processor for Toda language few-shot learning"""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
        
    def create_toda_few_shot_dataset(self, num_samples: int = 50) -> DatasetDict:
        """Create a few-shot dataset for Toda language"""
        
        # Sample Toda sentences with POS tags (transliterated)
        # Note: This is a demonstration with transliterated Toda
        # In practice, you would need actual Toda language data
        toda_samples = [
            {
                "tokens": ["nān", "ōr", "pōḍa", "kōḍa", "."],
                "pos_tags": ["PRON", "DET", "NOUN", "NOUN", "PUNCT"],
                "translation": "I am a Toda person."
            },
            {
                "tokens": ["nī", "ēṇḍa", "pōḍa", "kōḍa", "?"],
                "pos_tags": ["PRON", "PRON", "NOUN", "NOUN", "PUNCT"],
                "translation": "Are you a Toda person?"
            },
            {
                "tokens": ["nān", "pōḍa", "kōḍa", "āg", "irukān", "."],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PUNCT"],
                "translation": "I am being a Toda person."
            },
            {
                "tokens": ["nī", "ēṇḍa", "pōḍa", "kōḍa", "āg", "irukāy", "?"],
                "pos_tags": ["PRON", "PRON", "NOUN", "NOUN", "AUX", "VERB", "PUNCT"],
                "translation": "Are you being a Toda person?"
            },
            {
                "tokens": ["nān", "pōḍa", "kōḍa", "āg", "irukān", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Am I being a Toda person?"
            },
            {
                "tokens": ["nī", "pōḍa", "kōḍa", "āg", "irukāy", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Are you being a Toda person?"
            },
            {
                "tokens": ["nān", "pōḍa", "kōḍa", "āg", "irukān", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Am I being a Toda person?"
            },
            {
                "tokens": ["nī", "pōḍa", "kōḍa", "āg", "irukāy", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Are you being a Toda person?"
            },
            {
                "tokens": ["nān", "pōḍa", "kōḍa", "āg", "irukān", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Am I being a Toda person?"
            },
            {
                "tokens": ["nī", "pōḍa", "kōḍa", "āg", "irukāy", "ēṇḍa", "?"],
                "pos_tags": ["PRON", "NOUN", "NOUN", "AUX", "VERB", "PRON", "PUNCT"],
                "translation": "Are you being a Toda person?"
            }
        ]
        
        # Extend the dataset to the requested number of samples
        extended_samples = []
        for i in range(num_samples):
            sample = toda_samples[i % len(toda_samples)].copy()
            # Add some variation to make it more realistic
            sample["id"] = f"toda_{i:03d}"
            extended_samples.append(sample)
        
        # Split into train/validation/test
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        train_data = extended_samples[:train_size]
        val_data = extended_samples[train_size:train_size + val_size]
        test_data = extended_samples[train_size + val_size:]
        
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })
    
    def preprocess_toda_data(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess Toda dataset for few-shot learning"""
        
        def tokenize_and_align_labels(examples):
            """Tokenize and align labels with subword tokens"""
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                is_split_into_words=True
            )
            
            labels = []
            for i, label in enumerate(examples["pos_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self._get_label_id(label[word_idx]))
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        # Apply preprocessing to all splits
        processed_dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def _get_label_id(self, label: str) -> int:
        """Convert POS tag string to integer ID"""
        # Universal POS tags mapping
        pos_tag_mapping = {
            "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4,
            "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9,
            "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13,
            "SYM": 14, "VERB": 15, "X": 16
        }
        return pos_tag_mapping.get(label, 16)  # Default to "X" for unknown tags
    
    def create_toda_adapter_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Create dataset for Toda language adapter training"""
        
        def prepare_adapter_data(examples):
            """Prepare data for Toda language adapter training"""
            return {
                "input_ids": examples["input_ids"],
                "attention_mask": examples["attention_mask"],
                "tokens": examples.get("tokens", [])
            }
        
        adapter_dataset = dataset.map(
            prepare_adapter_data,
            batched=True,
            remove_columns=["labels"] if "labels" in dataset["train"].column_names else []
        )
        
        return adapter_dataset
    
    def augment_toda_data(self, dataset: DatasetDict, augmentation_factor: int = 2) -> DatasetDict:
        """Augment Toda dataset for better few-shot learning"""
        
        def augment_example(example):
            """Create augmented versions of an example"""
            augmented = []
            
            # Original example
            augmented.append(example)
            
            # Simple augmentation: add slight variations
            for i in range(augmentation_factor - 1):
                aug_example = example.copy()
                # Add some noise or variation here if needed
                # For now, just duplicate with a small modification
                aug_example["id"] = f"{example['id']}_aug_{i}"
                augmented.append(aug_example)
            
            return augmented
        
        # Apply augmentation to training data
        train_data = []
        for example in dataset["train"]:
            train_data.extend(augment_example(example))
        
        augmented_dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": dataset["validation"],
            "test": dataset["test"]
        })
        
        return augmented_dataset