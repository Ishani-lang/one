"""
Cross-lingual adapter model for POS tagging
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.adapters import AutoAdapterModel, AdapterConfig
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CrossLingualAdapterModel:
    """Cross-lingual adapter model for POS tagging"""
    
    def __init__(self, 
                 base_model_name: str = "xlm-roberta-base",
                 num_labels: int = 17,
                 adapter_config_name: str = "pfeiffer"):
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.adapter_config_name = adapter_config_name
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoAdapterModel.from_pretrained(base_model_name)
        
        # Add classification head
        self.classification_head = nn.Linear(
            self.model.config.hidden_size, 
            num_labels
        )
        
        # Adapter configuration
        self.adapter_config = AdapterConfig.load(adapter_config_name)
        
        # Language adapters
        self.language_adapters = {}
        self.task_adapters = {}
        
    def add_language_adapter(self, language: str, train_adapter: bool = True) -> None:
        """Add language-specific adapter"""
        try:
            self.model.add_adapter(language, config=self.adapter_config)
            self.language_adapters[language] = True
            
            if train_adapter:
                self.model.train_adapter([language])
                logger.info(f"Added and activated language adapter for {language}")
            else:
                logger.info(f"Added language adapter for {language}")
                
        except Exception as e:
            logger.error(f"Error adding language adapter for {language}: {e}")
            raise
    
    def add_task_adapter(self, task_name: str, train_adapter: bool = True) -> None:
        """Add task-specific adapter"""
        try:
            self.model.add_adapter(task_name, config=self.adapter_config)
            self.task_adapters[task_name] = True
            
            if train_adapter:
                self.model.train_adapter([task_name])
                logger.info(f"Added and activated task adapter for {task_name}")
            else:
                logger.info(f"Added task adapter for {task_name}")
                
        except Exception as e:
            logger.error(f"Error adding task adapter for {task_name}: {e}")
            raise
    
    def set_active_adapters(self, adapters: List[str]) -> None:
        """Set active adapters for inference"""
        try:
            self.model.set_active_adapters(adapters)
            logger.info(f"Set active adapters: {adapters}")
        except Exception as e:
            logger.error(f"Error setting active adapters {adapters}: {e}")
            raise
    
    def freeze_base_model(self) -> None:
        """Freeze base model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Frozen base model parameters")
    
    def unfreeze_base_model(self) -> None:
        """Unfreeze base model parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfrozen base model parameters")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        # Get model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply classification head
        logits = self.classification_head(hidden_states)
        
        result = {"logits": logits}
        
        if labels is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            # Only calculate loss on non-ignored labels
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            result["loss"] = loss
        
        return result
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        return predictions
    
    def save_adapters(self, save_path: str) -> None:
        """Save adapters to disk"""
        try:
            self.model.save_all_adapters(save_path)
            logger.info(f"Saved adapters to {save_path}")
        except Exception as e:
            logger.error(f"Error saving adapters to {save_path}: {e}")
            raise
    
    def load_adapters(self, load_path: str) -> None:
        """Load adapters from disk"""
        try:
            self.model.load_all_adapters(load_path)
            logger.info(f"Loaded adapters from {load_path}")
        except Exception as e:
            logger.error(f"Error loading adapters from {load_path}: {e}")
            raise
    
    def get_adapter_info(self) -> Dict[str, List[str]]:
        """Get information about available adapters"""
        return {
            "language_adapters": list(self.language_adapters.keys()),
            "task_adapters": list(self.task_adapters.keys()),
            "active_adapters": self.model.active_adapters
        }
    
    def train_adapter(self, adapter_name: str, 
                     train_dataset, 
                     eval_dataset,
                     training_args) -> None:
        """Train a specific adapter"""
        try:
            # Set active adapters
            self.model.set_active_adapters([adapter_name])
            
            # Create trainer
            from transformers import Trainer, TrainingArguments
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train the adapter
            trainer.train()
            
            logger.info(f"Trained adapter {adapter_name}")
            
        except Exception as e:
            logger.error(f"Error training adapter {adapter_name}: {e}")
            raise
    
    def evaluate_adapter(self, adapter_name: str, 
                        eval_dataset,
                        eval_args) -> Dict[str, float]:
        """Evaluate a specific adapter"""
        try:
            # Set active adapters
            self.model.set_active_adapters([adapter_name])
            
            # Create trainer
            from transformers import Trainer, TrainingArguments
            
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Evaluate the adapter
            eval_results = trainer.evaluate()
            
            logger.info(f"Evaluated adapter {adapter_name}: {eval_results}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error evaluating adapter {adapter_name}: {e}")
            raise