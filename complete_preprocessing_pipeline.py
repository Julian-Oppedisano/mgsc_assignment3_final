import argparse
import os
import logging
import yaml
import torch
import numpy as np
import random
import time
from pathlib import Path
from typing import Dict, Any

# Import custom modules
from preprocessing_pipeline import TextPreprocessor
from advanced_augmentation import AdvancedTextAugmenter
from oov_handling_and_integration import OOVHandler, DatasetCombiner
from preprocessing_dashboard import DatasetAnalyzerDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def create_directories(base_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for the pipeline.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'base': base_dir,
        'raw_data': os.path.join(base_dir, 'raw_data'),
        'processed_data': os.path.join(base_dir, 'processed_data'),
        'augmented_data': os.path.join(base_dir, 'augmented_data'),
        'analysis': os.path.join(base_dir, 'analysis'),
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return directories


def main():
    """
    Main function to run the complete preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description="Complete Text Classification Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--use_20newsgroups", action="store_true", dest="use_20newsgroups", help="Use 20 Newsgroups dataset")
    parser.add_argument("--no-use_20newsgroups", action="store_false", dest="use_20newsgroups", help="Don't use 20 Newsgroups dataset")
    parser.add_argument("--use_ag_news", action="store_true", dest="use_ag_news", help="Use AG News dataset")
    parser.add_argument("--no-use_ag_news", action="store_false", dest="use_ag_news", help="Don't use AG News dataset")
    parser.add_argument("--use_imdb", action="store_true", dest="use_imdb", help="Use IMDB dataset")
    parser.add_argument("--no-use_imdb", action="store_false", dest="use_imdb", help="Don't use IMDB dataset")
    parser.add_argument("--balance_classes", action="store_true", dest="balance_classes", help="Balance classes")
    parser.add_argument("--no-balance_classes", action="store_false", dest="balance_classes", help="Don't balance classes")
    parser.add_argument("--augment_data", action="store_true", dest="augment_data", help="Perform data augmentation")
    parser.add_argument("--no-augment_data", action="store_false", dest="augment_data", help="Don't perform data augmentation")
    parser.add_argument("--analyze_data", action="store_true", dest="analyze_data", help="Analyze the preprocessed data")
    parser.add_argument("--no-analyze_data", action="store_false", dest="analyze_data", help="Don't analyze the preprocessed data")
    parser.add_argument("--transformer_model", type=str, default=None, help="Transformer model for tokenization")
    
    # Set default values to None to differentiate between explicit False and not specified
    parser.set_defaults(
        use_20newsgroups=None,
        use_ag_news=None,
        use_imdb=None,
        balance_classes=None,
        augment_data=None,
        analyze_data=None
    )
    
    args = parser.parse_args()
    
    # Set up base configuration
    config = load_config(args.config)
    
    # Override config with command line arguments only if explicitly provided
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
        
    if args.seed is not None:
        config['seed'] = args.seed
        
    if args.use_20newsgroups is not None:
        config['use_20newsgroups'] = args.use_20newsgroups
        
    if args.use_ag_news is not None:
        config['use_ag_news'] = args.use_ag_news
        
    if args.use_imdb is not None:
        config['use_imdb'] = args.use_imdb
        
    if args.balance_classes is not None:
        config['balance_classes'] = args.balance_classes
        
    if args.augment_data is not None:
        config['augment_data'] = args.augment_data
        
    if args.analyze_data is not None:
        config['analyze_data'] = args.analyze_data
        
    if args.transformer_model is not None:
        config['transformer_model'] = args.transformer_model
    
    # Set random seeds for reproducibility
    set_random_seeds(config['seed'])
    
    # Create necessary directories
    directories = create_directories(config['output_dir'])
    
    # Start timing the pipeline
    start_time = time.time()
    
    # Step 1: Initialize the text preprocessor
    logger.info("Step 1: Initializing text preprocessor...")
    preprocessor = TextPreprocessor(
        transformer_model=config['transformer_model'],
        max_length=config.get('max_length', 512),
        do_stemming=config.get('do_stemming', True),
        do_lemmatization=config.get('do_lemmatization', False),
        remove_stopwords=config.get('remove_stopwords', True),
        remove_punctuation=config.get('remove_punctuation', True),
        lowercase=config.get('lowercase', True),
        random_seed=config['seed']
    )
    
    # Step 2: Initialize the dataset combiner
    logger.info("Step 2: Initializing dataset combiner...")
    dataset_combiner = DatasetCombiner(
        output_dir=directories['processed_data'],
        random_seed=config['seed'],
        max_samples_per_class=config.get('max_samples_per_class', None)
    )
    
    # Step 3: Combine datasets
    logger.info("Step 3: Combining datasets...")
    texts, labels, sources = dataset_combiner.combine_datasets(
        preprocessor=preprocessor,
        use_20newsgroups=config['use_20newsgroups'],
        use_ag_news=config['use_ag_news'],
        use_imdb=config['use_imdb'],
        balance_classes=config['balance_classes'],
        create_unified_labels=config.get('create_unified_labels', True)
    )
    
    # Step 4: Handle OOV words
    logger.info("Step 4: Handling OOV words...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['transformer_model'])
    
    oov_handler = OOVHandler(
        tokenizer=tokenizer,
        max_vocab_size=config.get('max_vocab_size', 30000),
        min_freq=config.get('min_freq', 5),
        handle_strategy=config.get('oov_strategy', 'subword')
    )
    
    processed_texts, processed_labels, processed_sources = dataset_combiner.apply_oov_handling(
        texts=texts,
        labels=labels,
        sources=sources,
        oov_handler=oov_handler
    )
    
    # Step 5: Augment data if requested
    if config['augment_data']:
        logger.info("Step 5: Augmenting data...")
        augmenter = AdvancedTextAugmenter(
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir=os.path.join(directories['base'], "model_cache"),
            random_seed=config['seed']
        )
        
        # Determine which augmentation techniques to use
        augmentation_techniques = config.get('augmentation_techniques', 
                                           ['synonym_replacement', 'random_insertion', 
                                            'random_deletion', 'random_swap'])
        
        # Add back_translation if specified
        if config.get('use_back_translation', False):
            augmentation_techniques.append('back_translation')
        
        augmented_texts, augmented_labels, augmented_sources = dataset_combiner.apply_augmentation(
            texts=processed_texts,
            labels=processed_labels,
            sources=processed_sources,
            augmenter=augmenter,
            augmentation_factor=config.get('augmentation_factor', 1),
            techniques=augmentation_techniques
        )
    else:
        logger.info("Skipping data augmentation")
        augmented_texts = processed_texts
        augmented_labels = processed_labels
        augmented_sources = processed_sources
    
    # Step 6: Split the dataset
    logger.info("Step 6: Splitting dataset...")
    split_data = dataset_combiner.split_dataset(
        texts=augmented_texts,
        labels=augmented_labels,
        sources=augmented_sources,
        test_size=config.get('test_size', 0.2),
        val_size=config.get('val_size', 0.1),
        stratify=config.get('stratify_splits', True)
    )
    
    # Step 7: Save the processed dataset
    logger.info("Step 7: Saving processed dataset...")
    dataset_combiner.save_dataset(split_data)
    
    # Step 8: Analyze the data if requested
    if config['analyze_data']:
        logger.info("Step 8: Analyzing processed data...")
        analyzer = DatasetAnalyzerDashboard(
            dataset_dir=directories['processed_data'],
            output_dir=directories['analysis'],
            transformer_model=config['transformer_model'],
            random_seed=config['seed']
        )
        
        analyzer.run_full_analysis()
    else:
        logger.info("Skipping data analysis")
    
    # Calculate and log the total processing time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Preprocessing pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return {
        'split_data': split_data,
        'directories': directories,
        'config': config,
        'processing_time': total_time
    }


if __name__ == "__main__":
    main()