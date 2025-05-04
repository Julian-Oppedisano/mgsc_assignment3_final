import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import pickle
import logging
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phase4_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_synthetic_results(test_df, num_classes, output_dir="output/phase4_demo"):
    """
    Generate synthetic model outputs to demonstrate Phase 4 visualization capabilities.
    
    Args:
        test_df: Test dataframe
        num_classes: Number of classes
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # True labels
    y_true = test_df['label'].values
    
    # Generate synthetic predictions (mostly correct but with some errors)
    # 80% correct predictions, 20% random
    y_pred = []
    for label in y_true:
        if random.random() < 0.80:  # 80% correct
            y_pred.append(label)
        else:
            # Random incorrect prediction
            incorrect_labels = [i for i in range(num_classes) if i != label]
            y_pred.append(random.choice(incorrect_labels))
    
    y_pred = np.array(y_pred)
    
    # Save predictions and true labels
    np.save(os.path.join(output_dir, "predictions.npy"), y_pred)
    np.save(os.path.join(output_dir, "true_labels.npy"), y_true)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Save report
    with open(os.path.join(output_dir, "classification_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save test results
    results = {
        'loss': 0.5,
        'accuracy': report['accuracy'],
        'f1': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'predictions': y_pred.tolist(),
        'true_labels': y_true.tolist()
    }
    
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return y_true, y_pred, report

def generate_confusion_matrix(y_true, y_pred, class_names, output_dir="output/phase4_demo"):
    """
    Generate and visualize confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory
    """
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
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    # Save confusion matrix
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    logger.info(f"Saved confusion matrix to {confusion_matrix_path}")
    
    # Also save raw confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

def visualize_performance_metrics(report, class_names, output_dir="output/phase4_demo"):
    """
    Visualize performance metrics (precision, recall, F1) per class.
    
    Args:
        report: Classification report
        class_names: List of class names
        output_dir: Output directory
    """
    # Extract per-class metrics
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for class_name, metrics in report.items():
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
    metrics_df.to_csv(os.path.join(output_dir, "class_metrics.csv"), index=False)
    
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
    sns.barplot(
        x="Class", 
        y="Value", 
        hue="Metric", 
        data=melted_df
    )
    
    plt.xticks(rotation=45, ha="right")
    plt.title("Performance Metrics per Class")
    plt.tight_layout()
    
    # Save plot
    metrics_path = os.path.join(output_dir, "class_metrics.png")
    plt.savefig(metrics_path)
    plt.close()
    
    logger.info(f"Saved performance metrics visualization to {metrics_path}")

def analyze_misclassifications(y_true, y_pred, class_names, output_dir="output/phase4_demo"):
    """
    Analyze misclassified examples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory
    """
    # Find misclassified examples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Count misclassifications per class
    true_classes = y_true[misclassified_indices]
    pred_classes = y_pred[misclassified_indices]
    
    # Create a DataFrame of misclassifications
    misclass_df = pd.DataFrame({
        "True Class": [class_names[i] for i in true_classes],
        "Predicted Class": [class_names[i] for i in pred_classes]
    })
    
    # Count misclassifications per class pair
    misclass_counts = misclass_df.groupby(["True Class", "Predicted Class"]).size().reset_index(name="Count")
    misclass_counts = misclass_counts.sort_values("Count", ascending=False)
    
    # Save misclassification counts
    misclass_counts.to_csv(os.path.join(output_dir, "misclassifications.csv"), index=False)
    
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
    plt.title("Top 20 Misclassifications")
    plt.tight_layout()
    
    # Save plot
    misclass_path = os.path.join(output_dir, "misclassifications.png")
    plt.savefig(misclass_path)
    plt.close()
    
    logger.info(f"Saved misclassification analysis to {misclass_path}")

def visualize_attention(tokenizer, text, output_dir="output/phase4_demo"):
    """
    Visualize a simplified attention mechanism as demonstration.
    
    Args:
        tokenizer: Tokenizer
        text: Input text
        output_dir: Output directory
    """
    # Tokenize input
    tokens = tokenizer.tokenize(text)
    
    # Generate synthetic attention weights (focused on important words)
    importance_words = ["movie", "film", "great", "good", "bad", "terrible", "amazing", "worst", "best",
                        "neural", "network", "computer", "algorithm", "model", "data", "learning",
                        "baseball", "team", "game", "player", "sport", "win", "score"]
    
    attention = np.zeros((len(tokens), len(tokens)))
    
    # Add synthetic patterns (just for demonstration)
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            # Diagonal attention (self-attention)
            if i == j:
                attention[i, j] = 0.5
            
            # Important words get more attention
            if token_i.lower() in importance_words:
                attention[i, j] += 0.2
            if token_j.lower() in importance_words:
                attention[i, j] += 0.2
                
            # Normalize
            attention[i, j] = min(attention[i, j], 1.0)
    
    # Visualize attention
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=False
    )
    
    plt.title(f"Simulated Attention Pattern")
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, "attention_example.png")
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Saved simulated attention visualization to {save_path}")

def main():
    """
    Main function to run Phase 4 demo.
    """
    logger.info("Starting Phase 4 demo...")
    
    # Create output directory
    output_dir = "output/phase4_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed data
    logger.info("Loading data...")
    data_dir = "output/processed_data"
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Sample a small subset for quick demo
    test_df = test_df.sample(n=500, random_state=42)
    
    # Load label encoder
    with open(os.path.join(data_dir, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    
    logger.info(f"Loaded {num_classes} classes")
    
    # Generate synthetic results
    logger.info("Generating synthetic model results...")
    y_true, y_pred, report = generate_synthetic_results(test_df, num_classes, output_dir)
    
    # a. Performance metrics visualization
    logger.info("Visualizing performance metrics...")
    visualize_performance_metrics(report, class_names, output_dir)
    
    # c. Confusion matrix
    logger.info("Generating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, class_names, output_dir)
    
    # Additional: Misclassification analysis
    logger.info("Analyzing misclassifications...")
    analyze_misclassifications(y_true, y_pred, class_names, output_dir)
    
    # b. Attention visualization (simplified demonstration)
    logger.info("Demonstrating attention visualization...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Example texts for different classes
    example_texts = [
        "The movie was amazing, great plot and fantastic acting",
        "Neural networks have significantly improved computer vision tasks",
        "The baseball team won the championship game with a record score"
    ]
    
    # Visualize attention for each example
    for i, text in enumerate(example_texts):
        visualize_attention(tokenizer, text, output_dir)
    
    logger.info("Phase 4 demo completed. Results saved to output/phase4_demo/")
    logger.info("Run 'python model_evaluation.py --model_dir output/phase4_demo' to test the full evaluation pipeline")

if __name__ == "__main__":
    main() 