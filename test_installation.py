#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.model_config import ModelConfig, TrainingConfig, UNIVERSAL_POS_TAGS
        print("✓ Config modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing config modules: {e}")
        return False
    
    try:
        from data.tamil_processor import TamilDataProcessor
        from data.toda_processor import TodaDataProcessor
        from data.utils import align_labels_with_tokens, compute_metrics
        print("✓ Data processing modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing data modules: {e}")
        return False
    
    try:
        from models.adapter_model import CrossLingualAdapterModel
        from models.pos_tagger import POSTagger
        print("✓ Model modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing model modules: {e}")
        return False
    
    try:
        from training.trainer import CrossLingualPOSTrainer, create_training_args
        from training.few_shot_trainer import FewShotLearningTrainer
        print("✓ Training modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing training modules: {e}")
        return False
    
    try:
        from evaluation.metrics import POSTaggingMetrics
        from evaluation.evaluator import CrossLingualEvaluator
        print("✓ Evaluation modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing evaluation modules: {e}")
        return False
    
    return True

def test_data_processing():
    """Test data processing functionality"""
    print("\nTesting data processing...")
    
    try:
        from data.tamil_processor import TamilDataProcessor
        from data.toda_processor import TodaDataProcessor
        
        # Test Tamil processor
        tamil_processor = TamilDataProcessor()
        tamil_dataset = tamil_processor.load_udpos_tamil()
        print(f"✓ Tamil dataset loaded: {len(tamil_dataset['train'])} samples")
        
        # Test Toda processor
        toda_processor = TodaDataProcessor()
        toda_dataset = toda_processor.create_toda_few_shot_dataset(num_samples=10)
        print(f"✓ Toda dataset created: {len(toda_dataset['train'])} samples")
        
        return True
    except Exception as e:
        print(f"✗ Error in data processing: {e}")
        return False

def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    
    try:
        from models.pos_tagger import POSTagger
        
        # Initialize POS tagger
        pos_tagger = POSTagger()
        print("✓ POS tagger initialized successfully")
        
        # Test model info
        model_info = pos_tagger.get_model_info()
        print(f"✓ Model info retrieved: {model_info['base_model']}")
        
        return True
    except Exception as e:
        print(f"✗ Error in model initialization: {e}")
        return False

def test_prediction():
    """Test basic prediction functionality"""
    print("\nTesting prediction functionality...")
    
    try:
        from models.pos_tagger import POSTagger
        
        # Initialize POS tagger
        pos_tagger = POSTagger()
        
        # Test prediction
        test_sentence = "This is a test sentence"
        predictions = pos_tagger.predict_pos_tags(test_sentence)
        print(f"✓ Prediction successful: {len(predictions)} tokens tagged")
        
        return True
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("TODA POS TAGGER - INSTALLATION TEST")
    print("="*60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test data processing
    if not test_data_processing():
        all_tests_passed = False
    
    # Test model initialization
    if not test_model_initialization():
        all_tests_passed = False
    
    # Test prediction
    if not test_prediction():
        all_tests_passed = False
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Run: python scripts/train_tamil_adapter.py")
        print("3. Run: python scripts/train_toda_fewshot.py")
    else:
        print("✗ SOME TESTS FAILED! Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version compatibility")
        print("3. Verify file permissions")
    print("="*60)

if __name__ == "__main__":
    main()