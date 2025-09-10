"""
Model configuration for Toda POS Tagger
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for the POS tagging model"""
    
    # Base model configuration
    base_model_name: str = "xlm-roberta-base"
    max_length: int = 512
    num_labels: int = 17  # Universal POS tags
    
    # Adapter configuration
    adapter_config_name: str = "pfeiffer"
    adapter_reduction_factor: int = 16
    adapter_non_linearity: str = "relu"
    adapter_dropout: float = 0.1
    
    # Language and task adapters
    language_adapters: List[str] = None
    task_adapters: List[str] = None
    
    # Cross-lingual transfer settings
    source_language: str = "ta"  # Tamil
    target_language: str = "toda"  # Toda
    
    def __post_init__(self):
        if self.language_adapters is None:
            self.language_adapters = ["ta", "toda"]
        if self.task_adapters is None:
            self.task_adapters = ["pos_tagging"]

@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Training parameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Few-shot learning parameters
    few_shot_samples: int = 100
    few_shot_epochs: int = 10
    few_shot_learning_rate: float = 1e-5
    
    # Evaluation parameters
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Output directories
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

# Universal POS tags mapping
UNIVERSAL_POS_TAGS = {
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

# Tamil-specific POS tags (if needed)
TAMIL_POS_TAGS = {
    "NOUN": "பெயர்ச்சொல்",
    "VERB": "வினைச்சொல்", 
    "ADJ": "பெயரடை",
    "ADV": "வினையுரிச்சொல்",
    "PRON": "பிரதிப்பெயர்",
    "DET": "குறிப்பிடுச்சொல்",
    "PREP": "முன்னிடைச்சொல்",
    "CONJ": "இணைப்புச்சொல்",
    "PUNCT": "நிறுத்தக்குறி"
}