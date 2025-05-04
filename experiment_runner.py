import os
import logging
import argparse
import importlib
import torch
import pandas as pd
import pickle
import json
from typing import Dict, Any
from transformer_trainer import TransformerTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment() -> bool:
    """
    Check if the environment is properly set up.
    
    Returns:
        True if the environment is set up, False otherwise
    """
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available. Using CPU for training (slower).")
    
    # Check for required files
    required_files = [
        "output/processed_data/train.csv",
        "output/processed_data/val.csv",
        "output/processed_data/test.csv",
        "output/processed_data/label_encoder.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("Please run the data preprocessing pipeline first.")
        return False
    
    # Check if the required modules are importable
    try:
        importlib.import_module("transformer_trainer")
        importlib.import_module("model_comparison")
        importlib.import_module("hyperparameter_tuning")
        importlib.import_module("attention_visualization")
    except ImportError as e:
        logger.error(f"Missing required module: {e}")
        return False
    
    logger.info("Environment check passed.")
    return True

def train_model(
    model_name: str,
    num_classes: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    config: dict
):
    """
    Train a model with the given configuration.
    
    Args:
        model_name: Name of the pre-trained transformer model
        num_classes: Number of classes
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Output directory
        config: Model configuration
    
    Returns:
        Test results
    """
    logger.info(f"Training {model_name} with the following configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create model directory
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = TransformerTrainer(
        model_name=model_name,
        num_classes=num_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=model_dir,
        max_length=config.get("max_length", 256),
        batch_size=config.get("batch_size", 16),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        num_epochs=config.get("num_epochs", 5),
        warmup_steps=config.get("warmup_steps", 0),
        dropout_rate=config.get("dropout_rate", 0.1),
        attention_dropout=config.get("attention_dropout", 0.1),
        use_mixed_precision=config.get("use_mixed_precision", True),
        patience=config.get("patience", 3),
        save_interval=config.get("save_interval", 1)
    )
    
    # Train model
    history = trainer.train()
    
    # Test model
    test_results = trainer.test()
    
    # Save configuration and results
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    with open(os.path.join(model_dir, "test_results.json"), "w") as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {
            k: float(v) if isinstance(v, (float, int)) or (hasattr(v, "item") and callable(getattr(v, "item"))) else v
            for k, v in test_results.items()
            if k not in ["predictions", "true_labels"]  # Exclude large arrays
        }
        json.dump(serializable_results, f, indent=2)
    
    return test_results

def run_phase3_experiments():
    """
    Run Phase 3 experiments with enhanced training features.
    """
    logger.info("Starting Phase 3 experiments")
    
    # Load the processed data
    data_dir = "output/processed_data"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Load label encoder to get number of classes
    with open(os.path.join(data_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Use a small subset for quick testing
    logger.info("Using a tiny subset (5%) for quick testing")
    train_df = train_df.sample(frac=0.05, random_state=42)
    val_df = val_df.sample(frac=0.05, random_state=42)
    test_df = test_df.sample(frac=0.05, random_state=42)
    
    # Output directory for Phase 3 models
    output_dir = "output/models/phase3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model configurations based on hyperparameter tuning results
    # Use minimal epochs and a smaller model for quick testing
    model_configs = {
        "distilbert-base-uncased": {
            "max_length": 128,     # Reduced max length
            "batch_size": 16,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "dropout_rate": 0.2,
            "attention_dropout": 0.1,
            "num_epochs": 2,       # Reduced epochs for quick testing
            "warmup_steps": 0,
            "use_mixed_precision": torch.cuda.is_available(),
            "patience": 3,
            "save_interval": 1
        }
    }
    
    # Train models with Phase 3 enhancements
    results = {}
    
    for model_name, config in model_configs.items():
        try:
            logger.info(f"Training {model_name} with Phase 3 enhancements")
            results[model_name] = train_model(
                model_name=model_name,
                num_classes=num_classes,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                output_dir=output_dir,
                config=config
            )
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    
    # Summarize results
    logger.info("Phase 3 Experiment Results:")
    for model_name, result in results.items():
        logger.info(f"{model_name} - Accuracy: {result.get('accuracy', 0):.4f}, F1: {result.get('f1', 0):.4f}")
    
    return results

def main():
    """
    Main function to run the experiment.
    """
    parser = argparse.ArgumentParser(description="Run Phase 3 experiments")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use ('cuda' or 'cpu'). Default is auto-detect.")
    args = parser.parse_args()
    
    # Set device
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.device == "cuda" else ""
    
    # Run experiments
    run_phase3_experiments()

if __name__ == "__main__":
    main()