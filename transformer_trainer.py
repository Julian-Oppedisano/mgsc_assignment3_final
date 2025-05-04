import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TransformerForClassification(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.1, attention_probs_dropout_prob=0.1):
        super(TransformerForClassification, self).__init__()
        
        # Load configuration and update dropout rates
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout_rate
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        
        # Load pre-trained model with custom configuration
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        
        # Get the hidden size from the model config
        hidden_size = config.hidden_size
        
        # Classification head with self-attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the hidden states
        sequence_output = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        
        # Apply self-attention pooling
        attention_weights = self.attention(sequence_output).transpose(1, 2)  # Shape: [batch_size, 1, seq_len]
        weighted_output = torch.bmm(attention_weights, sequence_output)  # Shape: [batch_size, 1, hidden_size]
        pooled_output = weighted_output.squeeze(1)  # Shape: [batch_size, hidden_size]
        
        # Apply classification head
        logits = self.classifier(pooled_output)
        
        return logits

class TransformerTrainer:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "models",
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 5,
        warmup_steps: int = 0,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        device: str = None,
        random_seed: int = 42,
        use_mixed_precision: bool = True,  # Add mixed precision parameter
        patience: int = 3,                 # Patience for early stopping
        save_interval: int = 0,            # Save model every N epochs (0 = only save best)
        checkpoint_dir: str = None         # Directory to save checkpoints
    ):
        """
        Initialize the transformer trainer.
        
        Args:
            model_name: Name of the pre-trained transformer model
            num_classes: Number of classes for classification
            train_df: Training data DataFrame with 'text' and 'label' columns
            val_df: Validation data DataFrame with 'text' and 'label' columns
            test_df: Test data DataFrame with 'text' and 'label' columns
            output_dir: Directory to save model checkpoints and results
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            dropout_rate: Dropout rate for regularization
            attention_dropout: Dropout rate for attention layers
            device: Device to use for training ('cuda' or 'cpu')
            random_seed: Random seed for reproducibility
            use_mixed_precision: Whether to use mixed precision training (faster on modern GPUs)
            patience: Number of epochs to wait before early stopping
            save_interval: Save model every N epochs (0 = only save best)
            checkpoint_dir: Directory to save checkpoints (if None, uses output_dir/checkpoints)
        """
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Set mixed precision flag
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        if self.use_mixed_precision:
            logger.info("Using mixed precision training (FP16)")
            self.scaler = GradScaler()
        else:
            logger.info("Using full precision training (FP32)")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize datasets
        self.train_dataset = TextClassificationDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        self.val_dataset = TextClassificationDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        self.test_dataset = TextClassificationDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            max_length
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize model
        self.model = TransformerForClassification(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            attention_probs_dropout_prob=attention_dropout
        )
        
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Other parameters
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Early stopping parameters
        self.patience = patience
        self.save_interval = save_interval
        
        # Checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        else:
            self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train(self) -> Dict:
        """
        Train the transformer model.
        
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    # Use mixed precision for forward pass
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(logits, labels)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    # Standard full-precision training
                    logits = self.model(input_ids, attention_mask)
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_results = self.evaluate(self.val_loader)
            history['val_loss'].append(val_results['loss'])
            history['val_accuracy'].append(val_results['accuracy'])
            history['val_f1'].append(val_results['f1'])
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_results['loss']:.4f}, Val Accuracy: {val_results['accuracy']:.4f}, Val F1: {val_results['f1']:.4f}")
            
            # Save checkpoint at specified intervals
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name.replace('/', '_')}_epoch{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_results['loss'],
                    'val_accuracy': val_results['accuracy'],
                    'val_f1': val_results['f1']
                }, checkpoint_path)
                logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            
            # Save the best model
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                early_stopping_counter = 0  # Reset counter when validation improves
                
                # Save model state dict
                model_path = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_best.pt")
                torch.save(self.model.state_dict(), model_path)
                
                # Also save a full checkpoint
                best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name.replace('/', '_')}_best.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_results['loss'],
                    'val_accuracy': val_results['accuracy'],
                    'val_f1': val_results['f1']
                }, best_checkpoint_path)
                
                logger.info(f"Saved best model to {model_path}")
            else:
                early_stopping_counter += 1
                logger.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{self.patience}")
                
                if early_stopping_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info("Training completed")
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        return history
    
    def evaluate(self, data_loader) -> Dict:
        """
        Evaluate the model on the given data loader.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def test(self) -> Dict:
        """
        Test the model on the test set.
        
        Returns:
            Dictionary with test metrics
        """
        logger.info("Testing the model on test set...")
        
        # Load the best model
        model_path = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_best.pt")
        self.model.load_state_dict(torch.load(model_path))
        
        # Evaluate on test set
        test_results = self.evaluate(self.test_loader)
        
        # Log results
        logger.info(f"Test Results - Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
        
        # Save detailed classification report
        class_report = classification_report(test_results['true_labels'], test_results['predictions'], output_dict=True)
        report_path = os.path.join(self.output_dir, f"{self.model_name.replace('/', '_')}_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=2)
            
        # Save predictions and true labels to file
        predictions_path = os.path.join(self.output_dir, "predictions.npy")
        true_labels_path = os.path.join(self.output_dir, "true_labels.npy")
        
        np.save(predictions_path, np.array(test_results['predictions']))
        np.save(true_labels_path, np.array(test_results['true_labels']))
        
        # Save test results to JSON
        test_results_path = os.path.join(self.output_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = {
                'loss': test_results['loss'],
                'accuracy': test_results['accuracy'],
                'f1': test_results['f1'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'predictions': test_results['predictions'],
                'true_labels': test_results['true_labels']
            }
            json.dump(results_json, f, indent=2)
        
        return test_results
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The epoch number of the checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Full checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint from epoch {epoch} with validation loss: {checkpoint['val_loss']:.4f}")
            return epoch
        else:
            # Model state dict only
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded model state dict")
            return 0

def main():
    """
    Main function to run model training and evaluation.
    """
    # Load the processed data
    data_dir = "output/processed_data"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Load label encoder to get number of classes
    with open(os.path.join(data_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Train BERT model
    bert_trainer = TransformerTrainer(
        model_name="bert-base-uncased",
        num_classes=num_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir="output/models/bert",
        max_length=256,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_epochs=3,
        warmup_steps=0,
        dropout_rate=0.1,
        attention_dropout=0.1
    )
    
    bert_history = bert_trainer.train()
    bert_test_results = bert_trainer.test()
    
    # Train RoBERTa model
    roberta_trainer = TransformerTrainer(
        model_name="roberta-base",
        num_classes=num_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir="output/models/roberta",
        max_length=256,
        batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_epochs=3,
        warmup_steps=0,
        dropout_rate=0.1,
        attention_dropout=0.1
    )
    
    roberta_history = roberta_trainer.train()
    roberta_test_results = roberta_trainer.test()
    
    # Train DistilBERT model
    distilbert_trainer = TransformerTrainer(
        model_name="distilbert-base-uncased",
        num_classes=num_classes,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir="output/models/distilbert",
        max_length=256,
        batch_size=16,
        learning_rate=5e-5,  # Slightly higher for DistilBERT
        weight_decay=0.01,
        num_epochs=4,  # Slightly more epochs for DistilBERT
        warmup_steps=0,
        dropout_rate=0.1,
        attention_dropout=0.1
    )
    
    distilbert_history = distilbert_trainer.train()
    distilbert_test_results = distilbert_trainer.test()
    
    # Compare models
    logger.info("Model Comparison:")
    logger.info(f"BERT - Test Accuracy: {bert_test_results['accuracy']:.4f}, F1: {bert_test_results['f1']:.4f}")
    logger.info(f"RoBERTa - Test Accuracy: {roberta_test_results['accuracy']:.4f}, F1: {roberta_test_results['f1']:.4f}")
    logger.info(f"DistilBERT - Test Accuracy: {distilbert_test_results['accuracy']:.4f}, F1: {distilbert_test_results['f1']:.4f}")

if __name__ == "__main__":
    main()