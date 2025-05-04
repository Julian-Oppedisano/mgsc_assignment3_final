import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# For explanations
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_explanation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextClassificationExplainer:
    def __init__(
        self,
        model_dir: str,
        output_dir: str = "output/explanations",
        device: str = None,
        label_encoder_path: str = "output/processed_data/label_encoder.pkl",
        max_length: int = 256
    ):
        """
        Initialize the model explainer.
        
        Args:
            model_dir: Directory containing the model files
            output_dir: Directory to save explanations
            device: Device to use for inference
            label_encoder_path: Path to the label encoder file
            max_length: Maximum sequence length for tokenization
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
        
        # Set max length
        self.max_length = max_length
        
        # Use distilbert-base-uncased as the base model
        base_model_name = "distilbert-base-uncased"
        
        # Load tokenizer
        try:
            # Try loading directly from model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            logger.info(f"Loaded tokenizer from {self.model_dir}")
        except:
            # Fall back to base pre-trained model
            logger.info(f"Loading tokenizer from {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load pretrained model with classification head
        try:
            # Try loading directly from model_dir
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                num_labels=len(self.class_names),
                local_files_only=True
            )
            logger.info(f"Loaded model from {self.model_dir}")
        except:
            # Fall back to base pre-trained model
            logger.info(f"Loading model from {base_model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=len(self.class_names)
            )
        
        # Load trained weights
        # First try the format model_name_best.pt
        model_path = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        
        # If that doesn't exist, try searching for any *_best.pt file
        if not os.path.exists(model_path):
            logger.info(f"Could not find model weights at {model_path}, searching for alternatives")
            best_pt_files = [f for f in os.listdir(self.model_dir) if f.endswith("_best.pt")]
            if best_pt_files:
                model_path = os.path.join(self.model_dir, best_pt_files[0])
                logger.info(f"Found alternative model weights at {model_path}")
            else:
                model_path = None
                logger.warning(f"No model weights found in {self.model_dir}, using base model")
        
        # Try to load the weights, but continue gracefully if it fails
        try:
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded model weights from {model_path}")
            else:
                logger.warning(f"Could not find model weights, using base model")
        except RuntimeError as e:
            logger.warning(f"Failed to load model weights due to incompatible structure: {e}")
            logger.warning("Using base model instead")
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions for a list of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Numpy array of predictions (probabilities for each class)
        """
        results = []
        batch_size = 8  # Process in small batches to avoid memory issues
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            results.append(probs)
        
        return np.vstack(results)

    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Predict class for a single text input.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_class_name, confidence)
        """
        probs = self.predict([text])[0]
        pred_class_idx = np.argmax(probs)
        pred_class = self.class_names[pred_class_idx]
        confidence = probs[pred_class_idx]
        
        return pred_class, confidence

    def explain_with_shap(self, text: str, num_features: int = 20) -> None:
        """
        Explain a prediction using SHAP.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in the explanation
        """
        logger.info(f"Generating SHAP explanation for: {text[:50]}...")
        
        # Get model prediction
        pred_class, confidence = self.predict_single(text)
        logger.info(f"Model predicts '{pred_class}' with {confidence:.4f} confidence")
        
        # Initialize the explainer
        # We'll use SHAP's Partition explainer for text which is more suitable for transformers
        # First, define a prediction function that returns probabilities
        def predict_fn(texts):
            return self.predict(texts)
        
        # Create masker that masks tokens
        masker = shap.maskers.Text(self.tokenizer)
        
        # Create the explainer
        explainer = shap.Explainer(predict_fn, masker, output_names=self.class_names)
        
        # Calculate SHAP values
        shap_values = explainer([text])
        
        # Create output directory
        shap_dir = os.path.join(self.output_dir, "shap")
        os.makedirs(shap_dir, exist_ok=True)
        
        # Plot the explanation
        plt.figure(figsize=(12, 6))
        shap.plots.text(shap_values, display=False)
        plt.title(f"SHAP Explanation - Predicted: {pred_class}")
        plt.tight_layout()
        
        # Save the explanation
        save_path = os.path.join(shap_dir, f"shap_explanation_{pred_class}.png")
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved SHAP explanation to {save_path}")
        
        # Also save a bar plot of feature importance
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values.mean(0), show=False)
        plt.title(f"SHAP Feature Importance - Predicted: {pred_class}")
        plt.tight_layout()
        
        # Save the importance plot
        importance_path = os.path.join(shap_dir, f"shap_importance_{pred_class}.png")
        plt.savefig(importance_path)
        plt.close()
        
        logger.info(f"Saved SHAP feature importance to {importance_path}")

    def explain_with_lime(self, text: str, num_features: int = 20) -> None:
        """
        Explain a prediction using LIME.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in the explanation
        """
        logger.info(f"Generating LIME explanation for: {text[:50]}...")
        
        # Get model prediction
        pred_class, confidence = self.predict_single(text)
        pred_class_idx = list(self.class_names).index(pred_class)
        logger.info(f"Model predicts '{pred_class}' with {confidence:.4f} confidence")
        
        try:
            # Initialize the LIME explainer
            explainer = LimeTextExplainer(class_names=self.class_names)
            
            # Define a prediction function that returns probabilities
            def predict_fn(texts):
                return self.predict(texts)
            
            # Generate the explanation
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=500
            )
            
            # Create output directory
            lime_dir = os.path.join(self.output_dir, "lime")
            os.makedirs(lime_dir, exist_ok=True)
            
            # Skip the matplotlib visualization to avoid errors
            # and focus on HTML and text output
            
            # Save a HTML visualization
            try:
                html_path = os.path.join(lime_dir, f"lime_explanation_{pred_class}_{hash(text) % 10000}.html")
                html = exp.as_html()
                with open(html_path, 'w') as f:
                    f.write(html)
                
                logger.info(f"Saved LIME HTML explanation to {html_path}")
                
                # Also save a simple text version of the explanation
                text_path = os.path.join(lime_dir, f"lime_explanation_{pred_class}_{hash(text) % 10000}.txt")
                with open(text_path, 'w') as f:
                    f.write(f"Explanation for: {text[:100]}...\n\n")
                    f.write(f"Predicted class: {pred_class} with {confidence:.4f} confidence\n\n")
                    f.write("Top features:\n")
                    for idx, (word, weight) in enumerate(exp.as_list(label=pred_class_idx)):
                        f.write(f"{idx+1}. {word}: {weight:.6f}\n")
                
                logger.info(f"Saved LIME text explanation to {text_path}")
            except Exception as e:
                logger.error(f"Error creating LIME outputs: {e}")
                
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            # Make sure to close any open figures
            plt.close('all')

    def explain_multiple_texts(self, texts: List[str], method: str = "both") -> None:
        """
        Generate explanations for multiple texts.
        
        Args:
            texts: List of texts to explain
            method: Method to use for explanation ('shap', 'lime', or 'both')
        """
        for i, text in enumerate(texts):
            logger.info(f"Explaining text {i+1}/{len(texts)}")
            
            # Use SHAP
            if method in ["shap", "both"]:
                try:
                    self.explain_with_shap(text)
                except Exception as e:
                    logger.error(f"Error generating SHAP explanation: {e}")
                    plt.close('all')  # Close any open figures
            
            # Use LIME
            if method in ["lime", "both"]:
                try:
                    self.explain_with_lime(text)
                except Exception as e:
                    logger.error(f"Error generating LIME explanation: {e}")
                    plt.close('all')  # Close any open figures

    def explain_test_examples(self, data_path: str, num_examples: int = 3, method: str = "both", max_texts: int = 0) -> None:
        """
        Explain examples from test data.
        
        Args:
            data_path: Path to the test data CSV
            num_examples: Number of examples per class to explain
            method: Method to use for explanation ('shap', 'lime', or 'both')
            max_texts: Maximum number of texts to explain (0 = no limit)
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Group by class
        grouped = data.groupby('label')
        
        # Get examples from each class
        examples = []
        for class_idx, group in grouped:
            # Get class name
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            logger.info(f"Getting examples for class {class_name}")
            
            # Get examples
            class_examples = group['text'].tolist()[:num_examples]
            examples.extend(class_examples)
            
            # Check if we've reached the maximum
            if max_texts > 0 and len(examples) >= max_texts:
                logger.info(f"Reached maximum number of texts ({max_texts}), stopping collection")
                examples = examples[:max_texts]
                break
        
        # Explain examples
        self.explain_multiple_texts(examples, method)

def main():
    """
    Main function to run model explanations.
    """
    parser = argparse.ArgumentParser(description="Generate explanations for model predictions")
    parser.add_argument("--model_dir", type=str, default="output/models/distilbert", 
                        help="Directory containing the model files")
    parser.add_argument("--output_dir", type=str, default="output/explanations",
                        help="Directory to save explanations")
    parser.add_argument("--test_data", type=str, default="output/processed_data/test.csv",
                        help="Path to test data for examples")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Number of examples per class to explain")
    parser.add_argument("--method", type=str, choices=["shap", "lime", "both"], default="both",
                        help="Explanation method to use")
    parser.add_argument("--custom_text", type=str, default=None,
                        help="Custom text to explain (optional)")
    parser.add_argument("--max_texts", type=int, default=0,
                        help="Maximum number of texts to explain (0 = no limit)")
    
    args = parser.parse_args()
    
    # Initialize explainer
    explainer = TextClassificationExplainer(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # If custom text is provided, explain it
    if args.custom_text:
        logger.info(f"Explaining custom text: {args.custom_text[:50]}...")
        explainer.explain_multiple_texts([args.custom_text], args.method)
    else:
        # Explain test examples
        logger.info(f"Explaining {args.num_examples} examples per class from {args.test_data}")
        explainer.explain_test_examples(
            data_path=args.test_data,
            num_examples=args.num_examples,
            method=args.method,
            max_texts=args.max_texts
        )
    
    logger.info("Explanation generation completed.")

if __name__ == "__main__":
    main() 