import os
import pickle
import pandas as pd
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformer_trainer import TransformerTrainer, TextClassificationDataset
from transformers import AdamW

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def grid_search(
    model_name: str,
    num_classes: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    learning_rates: List[float],
    dropout_rates: List[float],
    attention_dropouts: List[float],
    batch_sizes: List[int],
    weight_decays: List[float],
    max_length: int,
    num_epochs: int
) -> Tuple[Dict, List[Dict]]:
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        model_name: Name of the transformer model
        num_classes: Number of classes for classification
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        learning_rates: List of learning rates to try
        dropout_rates: List of dropout rates to try
        attention_dropouts: List of attention dropout rates to try
        batch_sizes: List of batch sizes to try
        weight_decays: List of weight decay values to try
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (best parameters, all results)
    """
    logger.info(f"Starting grid search for {model_name}...")
    results = []
    best_f1 = 0
    best_params = {}
    
    # Ensure output directory exists
    os.makedirs('output/hyperparameter_tuning', exist_ok=True)
    
    # Create a subset of train and val data for faster tuning if needed
    # We'll use the full data, but this could be commented out to use a subset
    # train_subset = train_df.sample(frac=0.2, random_state=42)
    # val_subset = val_df.sample(frac=0.2, random_state=42)
    train_subset = train_df
    val_subset = val_df
    
    # Create a dummy test set for the TransformerTrainer (we won't use it for tuning)
    test_df = val_subset.copy()
    
    # Iterate through all combinations
    for lr in learning_rates:
        for dropout in dropout_rates:
            for attn_dropout in attention_dropouts:
                for batch_size in batch_sizes:
                    for weight_decay in weight_decays:
                        # Log current hyperparameters
                        logger.info(f"Training with: lr={lr}, dropout={dropout}, "
                                   f"attn_dropout={attn_dropout}, batch_size={batch_size}, "
                                   f"weight_decay={weight_decay}")
                        
                        # Create model with current hyperparameters
                        output_dir = f"output/hyperparameter_tuning/{model_name}_lr{lr}_drop{dropout}_attn{attn_dropout}_bs{batch_size}_wd{weight_decay}"
                        
                        try:
                            trainer = TransformerTrainer(
                                model_name=model_name,
                                num_classes=num_classes,
                                train_df=train_subset,
                                val_df=val_subset,
                                test_df=test_df,
                                output_dir=output_dir,
                                max_length=max_length,
                                batch_size=batch_size,
                                learning_rate=lr,
                                weight_decay=weight_decay,
                                num_epochs=num_epochs,
                                dropout_rate=dropout,
                                attention_dropout=attn_dropout
                            )
                            
                            # Train model
                            history = trainer.train()
                            
                            # Evaluate on validation set
                            val_results = trainer.evaluate(trainer.val_loader)
                            
                            # Store results
                            result = {
                                'lr': lr,
                                'dropout': dropout,
                                'attn_dropout': attn_dropout,
                                'batch_size': batch_size,
                                'weight_decay': weight_decay,
                                'val_loss': val_results['loss'],
                                'val_accuracy': val_results['accuracy'],
                                'val_f1': val_results['f1'],
                                'val_precision': val_results['precision'],
                                'val_recall': val_results['recall']
                            }
                            
                            results.append(result)
                            
                            # Update best parameters if we have a better F1 score
                            if val_results['f1'] > best_f1:
                                best_f1 = val_results['f1']
                                best_params = result.copy()
                                logger.info(f"New best F1: {best_f1:.4f} with params: {best_params}")
                            
                        except Exception as e:
                            logger.error(f"Error with hyperparameters: {e}")
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'output/hyperparameter_tuning/{model_name}_all_results.csv', index=False)
    
    logger.info(f"Grid search completed for {model_name}")
    logger.info(f"Best F1: {best_f1:.4f} with params: {best_params}")
    
    return best_params, results

def main():
    """
    Main function to run hyperparameter tuning.
    """
    # Load the processed data
    data_dir = "output/processed_data"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    
    # Load label encoder to get number of classes
    with open(os.path.join(data_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Sample a subset of data for faster tuning
    logger.info("Using a subset of data (20%) for faster hyperparameter tuning")
    train_df = train_df.sample(frac=0.2, random_state=42)
    val_df = val_df.sample(frac=0.2, random_state=42)
    
    # We're focusing only on DistilBERT since it was the best performing model
    # from our initial evaluation, with 84.83% accuracy and 84.84% F1 score
    logger.info("Running hyperparameter tuning for DistilBERT only (best performing model)")
    logger.info("Initial DistilBERT performance: Accuracy: 84.83%, F1: 84.84%")
    
    # Run hyperparameter tuning for DistilBERT with more focused parameter ranges
    # based on our initial findings
    best_distilbert_params, distilbert_results = grid_search(
        model_name="distilbert-base-uncased",
        num_classes=num_classes,
        train_df=train_df,
        val_df=val_df,
        learning_rates=[5e-5, 7e-5, 1e-4],  # Focus around the successful learning rate
        dropout_rates=[0.1, 0.2],
        attention_dropouts=[0.1],
        batch_sizes=[16, 32],
        weight_decays=[0.01, 0.05],
        max_length=256,
        num_epochs=2  # Reduced for faster tuning
    )
    
    # Compare best results
    logger.info("Hyperparameter Tuning Results Summary:")
    logger.info(f"DistilBERT - Best F1: {best_distilbert_params['val_f1']:.4f} with params: {best_distilbert_params}")
    
    # Create a visualization of the hyperparameter tuning results
    visualize_tuning_results(distilbert_results, 'distilbert')
    
    # Add explanation of why we focused only on DistilBERT
    logger.info("\nRationale for focusing on DistilBERT:")
    logger.info("1. DistilBERT was clearly the best performer in initial testing (84.83% accuracy vs ~76% for BERT/RoBERTa)")
    logger.info("2. DistilBERT is much faster to train (about 38 min/epoch vs 83 min/epoch for BERT/RoBERTa)")
    logger.info("3. Given time constraints, focusing resources on the most promising model is more efficient")
    logger.info("4. DistilBERT is more lightweight, making it better for deployment scenarios")
    logger.info("5. The substantial performance gap (>8%) suggests architectural advantage rather than just hyperparameter differences")

def visualize_tuning_results(results: List[Dict], model_name: str) -> None:
    """
    Visualize hyperparameter tuning results.
    
    Args:
        results: List of result dictionaries
        model_name: Name of the model
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    os.makedirs('output/hyperparameter_tuning/plots', exist_ok=True)
    
    # Plot learning rate vs. F1 score
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='lr', y='val_f1', data=df)
    plt.title(f'{model_name} - Learning Rate vs. F1 Score')
    plt.tight_layout()
    plt.savefig(f'output/hyperparameter_tuning/plots/{model_name}_lr_vs_f1.png')
    plt.close()
    
    # Plot dropout rate vs. F1 score
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dropout', y='val_f1', data=df)
    plt.title(f'{model_name} - Dropout Rate vs. F1 Score')
    plt.tight_layout()
    plt.savefig(f'output/hyperparameter_tuning/plots/{model_name}_dropout_vs_f1.png')
    plt.close()
    
    # Plot batch size vs. F1 score
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='batch_size', y='val_f1', data=df)
    plt.title(f'{model_name} - Batch Size vs. F1 Score')
    plt.tight_layout()
    plt.savefig(f'output/hyperparameter_tuning/plots/{model_name}_batch_size_vs_f1.png')
    plt.close()
    
    # Plot weight decay vs. F1 score
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weight_decay', y='val_f1', data=df)
    plt.title(f'{model_name} - Weight Decay vs. F1 Score')
    plt.tight_layout()
    plt.savefig(f'output/hyperparameter_tuning/plots/{model_name}_weight_decay_vs_f1.png')
    plt.close()
    
    # Create a heatmap of learning rate vs. dropout rate
    pivot_df = df.pivot_table(
        values='val_f1',
        index='lr',
        columns='dropout',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title(f'{model_name} - Learning Rate vs. Dropout Rate (F1 Score)')
    plt.tight_layout()
    plt.savefig(f'output/hyperparameter_tuning/plots/{model_name}_lr_dropout_heatmap.png')
    plt.close()
    
    # Save top 5 combinations to CSV
    top_df = df.sort_values('val_f1', ascending=False).head(5)
    top_df.to_csv(f'output/hyperparameter_tuning/plots/{model_name}_top5_combinations.csv', index=False)

if __name__ == "__main__":
    main()