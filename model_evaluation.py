import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model_dir: str,
        output_dir: str = "output/evaluation",
        device: str = None,
        label_encoder_path: str = "output/processed_data/label_encoder.pkl"
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_dir: Directory containing the model files
            output_dir: Directory to save evaluation results
            device: Device to use for inference
            label_encoder_path: Path to the label encoder file
        """
        self.model_dir = model_dir
        self.model_name = os.path.basename(model_dir)
        
        # Create output directory
        self.output_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.class_names = self.label_encoder.classes_
        logger.info(f"Loaded {len(self.class_names)} classes")
        
        # Load test results
        self.test_results = self._load_test_results()
        
        # Load test report
        self.test_report = self._load_test_report()
    
    def _load_test_results(self) -> Dict:
        """Load test results from JSON file."""
        test_results_path = os.path.join(self.model_dir, "test_results.json")
        
        if not os.path.exists(test_results_path):
            logger.warning(f"Test results not found at {test_results_path}")
            return {}
        
        with open(test_results_path, 'r') as f:
            return json.load(f)
    
    def _load_test_report(self) -> Dict:
        """Load test report from JSON file."""
        # Find report file
        report_files = [f for f in os.listdir(self.model_dir) if f.endswith('_test_report.json')]
        
        if not report_files:
            logger.warning(f"No test report found in {self.model_dir}")
            return {}
        
        with open(os.path.join(self.model_dir, report_files[0]), 'r') as f:
            return json.load(f)
    
    def generate_confusion_matrix(self) -> None:
        """Generate and visualize confusion matrix."""
        # Check if we have predictions and true labels
        predictions_file = os.path.join(self.model_dir, "predictions.npy")
        true_labels_file = os.path.join(self.model_dir, "true_labels.npy")
        
        # If we don't have the files, try to get from test_df
        if not (os.path.exists(predictions_file) and os.path.exists(true_labels_file)):
            logger.warning("Predictions and true labels files not found")
            
            # Try to find predictions and true labels from test_results
            if 'predictions' not in self.test_results or 'true_labels' not in self.test_results:
                logger.error("Cannot generate confusion matrix: no predictions available")
                return
        
        # Get true labels and predictions
        if os.path.exists(predictions_file) and os.path.exists(true_labels_file):
            y_true = np.load(true_labels_file)
            y_pred = np.load(predictions_file)
        else:
            y_true = np.array(self.test_results['true_labels'])
            y_pred = np.array(self.test_results['predictions'])
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualize confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Normalized Confusion Matrix - {self.model_name}')
        plt.tight_layout()
        
        # Save confusion matrix
        confusion_matrix_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {confusion_matrix_path}")
        
        # Also save raw confusion matrix as CSV
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_df.to_csv(os.path.join(self.output_dir, "confusion_matrix.csv"))
    
    def visualize_performance_metrics(self) -> None:
        """Visualize performance metrics (precision, recall, F1) per class."""
        if not self.test_report:
            logger.error("Cannot visualize performance metrics: no test report available")
            return
        
        # Extract per-class metrics
        classes = []
        precision = []
        recall = []
        f1 = []
        
        for class_name, metrics in self.test_report.items():
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                classes.append(class_name)
                precision.append(metrics["precision"])
                recall.append(metrics["recall"])
                f1.append(metrics["f1-score"])
        
        # Create DataFrame
        metrics_df = pd.DataFrame({
            "Class": classes,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
        # Save metrics as CSV
        metrics_df.to_csv(os.path.join(self.output_dir, "class_metrics.csv"), index=False)
        
        # Visualize metrics
        plt.figure(figsize=(12, 8))
        
        # Melt DataFrame for seaborn
        melted_df = pd.melt(
            metrics_df, 
            id_vars=["Class"], 
            value_vars=["Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Value"
        )
        
        # Plot metrics
        g = sns.catplot(
            x="Class", 
            y="Value", 
            hue="Metric", 
            data=melted_df,
            kind="bar",
            height=6,
            aspect=2
        )
        
        g.set_xticklabels(rotation=45, ha="right")
        plt.title(f"Performance Metrics per Class - {self.model_name}")
        plt.tight_layout()
        
        # Save plot
        metrics_path = os.path.join(self.output_dir, "class_metrics.png")
        plt.savefig(metrics_path)
        plt.close()
        
        logger.info(f"Saved performance metrics visualization to {metrics_path}")
    
    def analyze_misclassifications(self) -> None:
        """Analyze misclassified examples."""
        # Check if we have predictions and true labels
        predictions_file = os.path.join(self.model_dir, "predictions.npy")
        true_labels_file = os.path.join(self.model_dir, "true_labels.npy")
        
        # If we don't have the files, we can't analyze misclassifications
        if not (os.path.exists(predictions_file) and os.path.exists(true_labels_file)):
            logger.warning("Cannot analyze misclassifications: predictions files not found")
            return
        
        # Load predictions and true labels
        y_true = np.load(true_labels_file)
        y_pred = np.load(predictions_file)
        
        # Find misclassified examples
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        # Count misclassifications per class
        true_classes = y_true[misclassified_indices]
        pred_classes = y_pred[misclassified_indices]
        
        # Create a DataFrame of misclassifications
        misclass_df = pd.DataFrame({
            "True Class": [self.class_names[i] for i in true_classes],
            "Predicted Class": [self.class_names[i] for i in pred_classes]
        })
        
        # Count misclassifications per class pair
        misclass_counts = misclass_df.groupby(["True Class", "Predicted Class"]).size().reset_index(name="Count")
        misclass_counts = misclass_counts.sort_values("Count", ascending=False)
        
        # Save misclassification counts
        misclass_counts.to_csv(os.path.join(self.output_dir, "misclassifications.csv"), index=False)
        
        # Visualize most common misclassifications
        top_misclass = misclass_counts.head(20)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            [f"{row['True Class']} → {row['Predicted Class']}" for _, row in top_misclass.iterrows()],
            top_misclass["Count"]
        )
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.5
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                    va='center')
        
        plt.xlabel("Count")
        plt.ylabel("Misclassification")
        plt.title(f"Top 20 Misclassifications - {self.model_name}")
        plt.tight_layout()
        
        # Save plot
        misclass_path = os.path.join(self.output_dir, "misclassifications.png")
        plt.savefig(misclass_path)
        plt.close()
        
        logger.info(f"Saved misclassification analysis to {misclass_path}")

class AttentionVisualizer:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        output_dir: str = "output/attention_visualization",
        device: str = None,
        label_encoder_path: str = "output/processed_data/label_encoder.pkl",
    ):
        """
        Initialize the attention visualizer.
        
        Args:
            model_name: Name of the pre-trained transformer model
            model_path: Path to the trained model
            output_dir: Directory to save visualizations
            device: Device to use for inference
            label_encoder_path: Path to the label encoder file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with attention output
        config = AutoConfig.from_pretrained(model_name)
        config.output_attentions = True
        
        self.model = AutoModel.from_pretrained(model_name, config=config)
        
        # Load trained model weights if available
        if os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Check if this is a full checkpoint or just the model state dict
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                # Filter out classifier weights from state_dict
                model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                # Load filtered state dict
                self.model.load_state_dict(model_state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.model_name = model_name
    
    def get_attention_weights(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get attention weights for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (attention_weights, tokens)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get attention weights (shape: [batch_size, num_heads, seq_len, seq_len])
        attention = outputs.attentions
        
        # Convert token IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get attention weights for the first example
        # Average attention weights across all layers and heads
        # First stack all layers, then take mean across layers and heads
        # This gives us a 2D attention matrix of shape [seq_len, seq_len]
        all_layers = torch.stack([attn[0] for attn in attention]) # shape: [num_layers, num_heads, seq_len, seq_len]
        # Average across both layers and heads dimensions
        avg_attention = all_layers.mean(dim=(0, 1))  # shape: [seq_len, seq_len]
        
        return avg_attention.cpu().numpy(), tokens
    
    def visualize_attention(self, text: str, example_name: str = "example") -> str:
        """
        Visualize attention weights for the given text.
        
        Args:
            text: Input text
            example_name: Name for the example
            
        Returns:
            Path to the saved visualization
        """
        # Get attention weights and tokens
        attention_weights, tokens = self.get_attention_weights(text)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Remove padding tokens for better visualization
        if "[PAD]" in tokens:
            pad_idx = tokens.index("[PAD]")
            attention_weights = attention_weights[:pad_idx, :pad_idx]
            tokens = tokens[:pad_idx]
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            annot=False
        )
        
        plt.title(f"Attention Weights - {example_name}")
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(self.output_dir, f"{example_name.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved attention visualization to {save_path}")
        
        return save_path
    
    def visualize_class_examples(self, data_path: str, num_examples: int = 3) -> None:
        """
        Visualize attention patterns for examples from each class.
        
        Args:
            data_path: Path to the data file (CSV)
            num_examples: Number of examples to visualize per class
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Group by class
        grouped = data.groupby('label')
        
        # Create a directory for each class
        for class_idx, group in grouped:
            # Get class name
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            
            # Create class directory
            class_dir = os.path.join(self.output_dir, f"class_{class_idx}_{class_name.replace(' ', '_')}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Get examples
            examples = group['text'].tolist()[:num_examples]
            
            # Visualize attention for each example
            for i, example in enumerate(examples):
                try:
                    # Get attention weights and tokens
                    attention_weights, tokens = self.get_attention_weights(example)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 10))
                    
                    # Remove padding tokens for better visualization
                    if "[PAD]" in tokens:
                        pad_idx = tokens.index("[PAD]")
                        attention_weights = attention_weights[:pad_idx, :pad_idx]
                        tokens = tokens[:pad_idx]
                    
                    # Plot heatmap
                    sns.heatmap(
                        attention_weights,
                        xticklabels=tokens,
                        yticklabels=tokens,
                        cmap="viridis",
                        annot=False
                    )
                    
                    plt.title(f"Attention Weights - Class {class_name} - Example {i+1}")
                    plt.tight_layout()
                    
                    # Save the figure
                    save_path = os.path.join(class_dir, f"example_{i+1}.png")
                    plt.savefig(save_path)
                    plt.close()
                    
                    logger.info(f"Saved attention visualization for class {class_name}, example {i+1}")
                    
                except Exception as e:
                    logger.error(f"Error visualizing example {i+1} for class {class_name}: {e}")

def main():
    """
    Main function to run model evaluation and interpretation.
    """
    parser = argparse.ArgumentParser(description="Evaluate and interpret transformer models")
    parser.add_argument("--model_dir", type=str, default="output/models/distilbert", 
                        help="Directory containing the model files")
    parser.add_argument("--output_dir", type=str, default="output/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--attention", action="store_true",
                        help="Whether to visualize attention weights")
    parser.add_argument("--test_data", type=str, default="output/processed_data/test.csv",
                        help="Path to test data for attention visualization")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples per class for attention visualization")
    
    args = parser.parse_args()
    
    # Step 1: Model Evaluation
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Generate performance metrics
    logger.info("Generating performance metrics visualizations...")
    evaluator.visualize_performance_metrics()
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix...")
    evaluator.generate_confusion_matrix()
    
    # Analyze misclassifications
    logger.info("Analyzing misclassifications...")
    evaluator.analyze_misclassifications()
    
    # Step 2: Attention Visualization (optional)
    if args.attention:
        logger.info("Visualizing attention weights...")
        
        # Find the model file
        model_path = os.path.join(args.model_dir, f"{os.path.basename(args.model_dir)}_best.pt")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return
        
        # Initialize attention visualizer
        visualizer = AttentionVisualizer(
            model_name=os.path.basename(args.model_dir),
            model_path=model_path,
            output_dir=os.path.join(args.output_dir, os.path.basename(args.model_dir), "attention")
        )
        
        # Visualize attention for examples from each class
        visualizer.visualize_class_examples(
            data_path=args.test_data,
            num_examples=args.num_examples
        )
    
    logger.info("Model evaluation and interpretation completed.")

if __name__ == "__main__":
    main() 