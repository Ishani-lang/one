# Toda POS Tagger - Usage Guide

This guide provides comprehensive instructions for using the Toda POS Tagger with cross-lingual adapter-based transfer learning.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run example to verify installation
python example_usage.py
```

### 2. Training Pipeline

#### Step 1: Train Tamil Language and Task Adapters

```bash
python scripts/train_tamil_adapter.py
```

This will:
- Load Tamil POS tagging dataset from Hugging Face
- Train Tamil language adapter
- Train POS tagging task adapter
- Evaluate on test set
- Save trained adapters to `./outputs/tamil_pos_tagger/`

#### Step 2: Train Toda Language Adapter with Few-Shot Learning

```bash
python scripts/train_toda_fewshot.py
```

This will:
- Create Toda few-shot dataset
- Load pre-trained Tamil adapters
- Train Toda language adapter
- Perform cross-lingual transfer
- Evaluate performance
- Save complete model to `./outputs/toda_pos_tagger/`

### 3. Evaluation

```bash
# Evaluate Tamil model
python scripts/evaluate.py --tamil-model ./outputs/tamil_pos_tagger/

# Evaluate Toda model
python scripts/evaluate.py --toda-model ./outputs/toda_pos_tagger/

# Compare both models
python scripts/evaluate.py --tamil-model ./outputs/tamil_pos_tagger/ --toda-model ./outputs/toda_pos_tagger/ --compare
```

### 4. Inference

#### Interactive Mode
```bash
python scripts/inference.py --model-path ./outputs/toda_pos_tagger/ --mode interactive
```

#### Batch Mode
```bash
# Create input file with sentences
echo "nān ōr pōḍa kōḍa āg irukān" > input.txt
echo "nī ēṇḍa pōḍa kōḍa āg irukāy" >> input.txt

# Run batch inference
python scripts/inference.py --model-path ./outputs/toda_pos_tagger/ --mode batch --input-file input.txt --output-file output.txt
```

#### Demo Mode
```bash
python scripts/inference.py --model-path ./outputs/toda_pos_tagger/ --mode demo
```

#### Single Sentence
```bash
python scripts/inference.py --model-path ./outputs/toda_pos_tagger/ --sentence "nān ōr pōḍa kōḍa āg irukān"
```

## Detailed Usage

### Configuration

The model configuration can be modified in `config/model_config.py`:

```python
@dataclass
class ModelConfig:
    base_model_name: str = "xlm-roberta-base"  # Base multilingual model
    num_labels: int = 17                       # Number of POS tags
    adapter_config_name: str = "pfeiffer"      # Adapter configuration
    source_language: str = "ta"                # Source language (Tamil)
    target_language: str = "toda"              # Target language (Toda)
```

### Data Processing

#### Tamil Data
- Automatically loads from Hugging Face datasets
- Falls back to sample data if UD POS dataset unavailable
- Preprocesses for tokenization and label alignment

#### Toda Data
- Creates synthetic few-shot dataset
- Supports data augmentation
- Includes transliterated Toda sentences

### Model Architecture

The POS tagger uses:
- **Base Model**: XLM-RoBERTa (multilingual)
- **Language Adapters**: Tamil and Toda specific
- **Task Adapter**: POS tagging task
- **Classification Head**: Maps to 17 Universal POS tags

### Training Process

1. **Tamil Adapter Training**:
   - Train Tamil language adapter on Tamil data
   - Train POS task adapter on Tamil data
   - Combine adapters for Tamil POS tagging

2. **Toda Few-Shot Training**:
   - Load pre-trained Tamil adapters
   - Train Toda language adapter on few-shot data
   - Perform cross-lingual transfer

### Evaluation Metrics

- **Accuracy**: Token-level accuracy
- **F1 Score**: Weighted F1 score across all POS tags
- **Precision**: Weighted precision
- **Recall**: Weighted recall
- **Sequence Accuracy**: Exact sequence match accuracy

### Adapter Management

```python
# Get model information
model_info = pos_tagger.get_model_info()
print(model_info)

# Set active adapters
pos_tagger.adapter_model.set_active_adapters(["ta", "pos_tagging"])

# Save adapters
pos_tagger.save_model("./my_model/")

# Load adapters
pos_tagger.load_model("./my_model/")
```

## Advanced Usage

### Custom Datasets

To use your own datasets:

```python
from data.tamil_processor import TamilDataProcessor

# Create custom dataset
custom_dataset = {
    "tokens": [["word1", "word2", "word3"]],
    "pos_tags": [["NOUN", "VERB", "PUNCT"]]
}

# Process dataset
processor = TamilDataProcessor()
processed = processor.preprocess_tamil_data(custom_dataset)
```

### Few-Shot Learning

For custom few-shot scenarios:

```python
from training.few_shot_trainer import FewShotLearningTrainer

# Create few-shot trainer
trainer = FewShotLearningTrainer(
    model=pos_tagger.adapter_model,
    tokenizer=pos_tagger.adapter_model.tokenizer,
    training_args=training_args,
    base_adapters=["ta", "pos_tagging"]
)

# Train few-shot adapter
results = trainer.train_few_shot_adapter(
    adapter_name="custom_language",
    few_shot_dataset=your_dataset
)
```

### Cross-Lingual Transfer

```python
# Perform cross-lingual transfer
transfer_results = trainer.cross_lingual_transfer(
    source_language="ta",
    target_language="custom_lang",
    target_dataset=target_data
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training arguments
   - Use gradient accumulation
   - Use smaller base model

2. **Dataset Loading Errors**:
   - Check internet connection
   - Verify Hugging Face dataset availability
   - Use fallback sample data

3. **Adapter Loading Errors**:
   - Ensure model path exists
   - Check adapter compatibility
   - Verify model configuration

### Performance Optimization

1. **Training Speed**:
   - Use mixed precision training
   - Increase batch size if memory allows
   - Use gradient checkpointing

2. **Memory Usage**:
   - Use smaller base models
   - Reduce sequence length
   - Use model parallelism

## File Structure

```
toda_pos_tagger/
├── config/                 # Configuration files
├── data/                   # Data processing modules
├── models/                 # Model implementations
├── training/               # Training utilities
├── evaluation/             # Evaluation tools
├── scripts/                # Main scripts
├── notebooks/              # Jupyter notebooks
├── outputs/                # Model outputs
└── requirements.txt        # Dependencies
```

## API Reference

### POSTagger Class

```python
class POSTagger:
    def __init__(self, base_model_name, num_labels, adapter_config_name)
    def setup_tamil_adapters(self)
    def setup_toda_adapters(self)
    def train_tamil_adapters(self, train_dataset, eval_dataset, training_args)
    def train_toda_few_shot(self, toda_dataset, training_args)
    def predict_pos_tags(self, text, return_tokens=True)
    def evaluate(self, eval_dataset)
    def save_model(self, save_path)
    def load_model(self, load_path)
```

### Data Processors

```python
class TamilDataProcessor:
    def load_udpos_tamil(self)
    def preprocess_tamil_data(self, dataset)
    def create_tamil_adapter_dataset(self, dataset)

class TodaDataProcessor:
    def create_toda_few_shot_dataset(self, num_samples)
    def preprocess_toda_data(self, dataset)
    def augment_toda_data(self, dataset, augmentation_factor)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{toda_pos_tagger,
  title={Toda POS Tagger: Cross-lingual Adapter-based Transfer Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/toda-pos-tagger}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example notebooks