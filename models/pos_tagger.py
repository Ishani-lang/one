"""
POS Tagger model with cross-lingual transfer learning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.adapters import AutoAdapterModel, AdapterConfig
from typing import Dict, List, Optional, Tuple, Union
import logging
from .adapter_model import CrossLingualAdapterModel

logger = logging.getLogger(__name__)

class POSTagger:
    """POS Tagger with cross-lingual adapter support"""
    
    def __init__(self, 
                 base_model_name: str = "xlm-roberta-base",
                 num_labels: int = 17,
                 adapter_config_name: str = "pfeiffer"):
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.adapter_config_name = adapter_config_name
        
        # Initialize the adapter model
        self.adapter_model = CrossLingualAdapterModel(
            base_model_name=base_model_name,
            num_labels=num_labels,
            adapter_config_name=adapter_config_name
        )
        
        # Label mapping
        self.label_mapping = {
            0: "ADJ", 1: "ADP", 2: "ADV", 3: "AUX", 4: "CCONJ",
            5: "DET", 6: "INTJ", 7: "NOUN", 8: "NUM", 9: "PART",
            10: "PRON", 11: "PROPN", 12: "PUNCT", 13: "SCONJ",
            14: "SYM", 15: "VERB", 16: "X"
        }
        
        self.id2label = {v: k for k, v in self.label_mapping.items()}
        
    def setup_tamil_adapters(self) -> None:
        """Setup Tamil language and POS task adapters"""
        try:
            # Add Tamil language adapter
            self.adapter_model.add_language_adapter("ta", train_adapter=True)
            
            # Add POS tagging task adapter
            self.adapter_model.add_task_adapter("pos_tagging", train_adapter=True)
            
            # Set active adapters
            self.adapter_model.set_active_adapters(["ta", "pos_tagging"])
            
            logger.info("Setup Tamil language and POS task adapters")
            
        except Exception as e:
            logger.error(f"Error setting up Tamil adapters: {e}")
            raise
    
    def setup_toda_adapters(self) -> None:
        """Setup Toda language adapter for few-shot learning"""
        try:
            # Add Toda language adapter
            self.adapter_model.add_language_adapter("toda", train_adapter=True)
            
            # Set active adapters (Toda + POS task)
            self.adapter_model.set_active_adapters(["toda", "pos_tagging"])
            
            logger.info("Setup Toda language adapter for few-shot learning")
            
        except Exception as e:
            logger.error(f"Error setting up Toda adapters: {e}")
            raise
    
    def train_tamil_adapters(self, 
                           train_dataset,
                           eval_dataset,
                           training_args) -> None:
        """Train Tamil language and POS task adapters"""
        try:
            # Setup Tamil adapters
            self.setup_tamil_adapters()
            
            # Train language adapter first
            logger.info("Training Tamil language adapter...")
            self.adapter_model.train_adapter("ta", train_dataset, eval_dataset, training_args)
            
            # Train task adapter
            logger.info("Training POS tagging task adapter...")
            self.adapter_model.train_adapter("pos_tagging", train_dataset, eval_dataset, training_args)
            
            logger.info("Completed training Tamil adapters")
            
        except Exception as e:
            logger.error(f"Error training Tamil adapters: {e}")
            raise
    
    def train_toda_few_shot(self, 
                          toda_dataset,
                          training_args) -> None:
        """Train Toda language adapter with few-shot learning"""
        try:
            # Setup Toda adapters
            self.setup_toda_adapters()
            
            # Train Toda language adapter with few-shot data
            logger.info("Training Toda language adapter with few-shot learning...")
            self.adapter_model.train_adapter("toda", toda_dataset, None, training_args)
            
            logger.info("Completed training Toda adapter with few-shot learning")
            
        except Exception as e:
            logger.error(f"Error training Toda few-shot: {e}")
            raise
    
    def predict_pos_tags(self, 
                        text: Union[str, List[str]], 
                        return_tokens: bool = True) -> Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]:
        """Predict POS tags for input text"""
        try:
            if isinstance(text, str):
                # Single sentence
                return self._predict_single(text, return_tokens)
            else:
                # Multiple sentences
                return [self._predict_single(sentence, return_tokens) for sentence in text]
                
        except Exception as e:
            logger.error(f"Error predicting POS tags: {e}")
            raise
    
    def _predict_single(self, text: str, return_tokens: bool = True) -> List[Tuple[str, str]]:
        """Predict POS tags for a single sentence"""
        # Tokenize input
        inputs = self.adapter_model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.adapter_model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            predictions = torch.argmax(outputs["logits"], dim=-1)
        
        # Convert to tokens and tags
        tokens = self.adapter_model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_tags = [self.label_mapping[pred.item()] for pred in predictions[0]]
        
        # Filter out special tokens
        filtered_results = []
        for token, tag in zip(tokens, pred_tags):
            if token not in ["<s>", "</s>", "<pad>", "<unk>"]:
                if return_tokens:
                    filtered_results.append((token, tag))
                else:
                    filtered_results.append(tag)
        
        return filtered_results
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset"""
        try:
            # Create evaluation function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                
                # Remove ignored labels (-100)
                predictions = torch.argmax(torch.tensor(predictions), dim=2)
                
                # Flatten predictions and labels
                true_predictions = []
                true_labels = []
                
                for prediction, label in zip(predictions, labels):
                    for pred, lab in zip(prediction, label):
                        if lab != -100:
                            true_predictions.append(pred.item())
                            true_labels.append(lab)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                
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
            
            # Evaluate using the adapter model
            eval_results = self.adapter_model.evaluate_adapter(
                adapter_name="pos_tagging",
                eval_dataset=eval_dataset,
                eval_args=None  # Will be set by the trainer
            )
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, save_path: str) -> None:
        """Save the complete model"""
        try:
            # Save adapters
            self.adapter_model.save_adapters(save_path)
            
            # Save tokenizer
            self.adapter_model.tokenizer.save_pretrained(save_path)
            
            # Save model configuration
            import json
            config = {
                "base_model_name": self.base_model_name,
                "num_labels": self.num_labels,
                "adapter_config_name": self.adapter_config_name,
                "label_mapping": self.label_mapping
            }
            
            with open(f"{save_path}/config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved model to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, load_path: str) -> None:
        """Load the complete model"""
        try:
            # Load adapters
            self.adapter_model.load_adapters(load_path)
            
            # Load tokenizer
            self.adapter_model.tokenizer = AutoTokenizer.from_pretrained(load_path)
            
            # Load model configuration
            import json
            with open(f"{load_path}/config.json", "r") as f:
                config = json.load(f)
            
            self.label_mapping = config["label_mapping"]
            self.id2label = {v: k for k, v in self.label_mapping.items()}
            
            logger.info(f"Loaded model from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "base_model": self.base_model_name,
            "num_labels": self.num_labels,
            "adapter_config": self.adapter_config_name,
            "adapters": self.adapter_model.get_adapter_info(),
            "label_mapping": self.label_mapping
        }