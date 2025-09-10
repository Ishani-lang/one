"""
Tamil data processing for POS tagging
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re
from .utils import align_labels_with_tokens

class TamilDataProcessor:
    """Data processor for Tamil POS tagging datasets"""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
        
    def load_udpos_tamil(self) -> DatasetDict:
        """Load Tamil UD POS dataset from Hugging Face"""
        try:
            from datasets import load_dataset
            # Try to load the UD POS dataset for Tamil
            dataset = load_dataset("universal_dependencies", "ta_ttb")
            return dataset
        except Exception as e:
            print(f"Error loading UD POS Tamil dataset: {e}")
            # Fallback to a different Tamil dataset
            return self._load_fallback_tamil_dataset()
    
    def _load_fallback_tamil_dataset(self) -> DatasetDict:
        """Load fallback Tamil dataset if UD POS is not available"""
        # Create a sample Tamil dataset for demonstration
        sample_data = {
            "tokens": [
                ["நான்", "ஒரு", "மாணவன்", "."],
                ["அவள்", "பள்ளிக்கு", "போகிறாள்", "."],
                ["இது", "ஒரு", "நல்ல", "புத்தகம்", "."],
                ["நாங்கள்", "வீட்டில்", "இருக்கிறோம்", "."],
                ["அவர்கள்", "விளையாடுகிறார்கள்", "."]
            ],
            "pos_tags": [
                ["PRON", "DET", "NOUN", "PUNCT"],
                ["PRON", "NOUN", "VERB", "PUNCT"],
                ["PRON", "DET", "ADJ", "NOUN", "PUNCT"],
                ["PRON", "NOUN", "VERB", "PUNCT"],
                ["PRON", "VERB", "PUNCT"]
            ]
        }
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_dict(sample_data)
        return DatasetDict({
            "train": dataset,
            "validation": dataset,
            "test": dataset
        })
    
    def preprocess_tamil_data(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess Tamil dataset for training"""
        
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
    
    def create_tamil_adapter_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Create dataset specifically for training Tamil language adapter"""
        
        def prepare_adapter_data(examples):
            """Prepare data for language adapter training"""
            # For language adapter, we mainly need the input text
            # The task-specific labels will be added later
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
    
    def get_tamil_sample_data(self, num_samples: int = 100) -> List[Dict]:
        """Get sample Tamil data for few-shot learning"""
        sample_sentences = [
            {
                "tokens": ["நான்", "ஒரு", "மாணவன்", "."],
                "pos_tags": ["PRON", "DET", "NOUN", "PUNCT"]
            },
            {
                "tokens": ["அவள்", "பள்ளிக்கு", "போகிறாள்", "."],
                "pos_tags": ["PRON", "NOUN", "VERB", "PUNCT"]
            },
            {
                "tokens": ["இது", "ஒரு", "நல்ல", "புத்தகம்", "."],
                "pos_tags": ["PRON", "DET", "ADJ", "NOUN", "PUNCT"]
            },
            {
                "tokens": ["நாங்கள்", "வீட்டில்", "இருக்கிறோம்", "."],
                "pos_tags": ["PRON", "NOUN", "VERB", "PUNCT"]
            },
            {
                "tokens": ["அவர்கள்", "விளையாடுகிறார்கள்", "."],
                "pos_tags": ["PRON", "VERB", "PUNCT"]
            }
        ]
        
        return sample_sentences[:num_samples]