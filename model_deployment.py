import os
import torch
import json
import argparse
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pickle
from typing import Dict, List, Optional, Tuple, Union

# For ONNX export
import onnx
import onnxruntime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(
        self,
        model_dir: str,
        output_dir: str = "output/deployment",
        device: str = None,
        label_encoder_path: str = "output/processed_data/label_encoder.pkl",
        max_length: int = 256
    ):
        """
        Initialize the model deployer.
        
        Args:
            model_dir: Directory containing the model files
            output_dir: Directory to save deployment artifacts
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

    def export_to_onnx(self, onnx_path: Optional[str] = None) -> str:
        """
        Export the PyTorch model to ONNX format.
        
        Args:
            onnx_path: Path to save the ONNX model (optional)
            
        Returns:
            Path to the exported ONNX model
        """
        if onnx_path is None:
            onnx_path = os.path.join(self.output_dir, f"{self.model_name}_model.onnx")
        
        logger.info(f"Exporting model to ONNX format: {onnx_path}")
        
        # Create a dummy input
        dummy_input_ids = torch.ones(1, self.max_length, dtype=torch.long).to(self.device)
        dummy_attention_mask = torch.ones(1, self.max_length, dtype=torch.long).to(self.device)
        dummy_inputs = {
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask
        }
        
        # Export the model
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy_inputs,),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=14,
                export_params=True
            )
        
        # Verify the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model exported and verified: {onnx_path}")
        
        # Create inference configuration file
        inference_config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_classes": len(self.class_names),
            "class_names": self.class_names.tolist(),
            "model_path": onnx_path
        }
        
        config_path = os.path.join(os.path.dirname(onnx_path), "inference_config.json")
        with open(config_path, 'w') as f:
            json.dump(inference_config, f, indent=2)
        
        logger.info(f"Inference configuration saved to {config_path}")
        
        # Save tokenizer for inference
        tokenizer_path = os.path.join(os.path.dirname(onnx_path), "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        return onnx_path

    def test_onnx_inference(self, onnx_path: str, text: str) -> Dict:
        """
        Test inference with the exported ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            text: Text to run inference on
            
        Returns:
            Dictionary with inference results
        """
        logger.info(f"Testing ONNX inference with text: {text[:50]}...")
        
        # Create ONNX inference session
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = onnxruntime.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Run inference
        ort_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
        
        # Get model outputs
        ort_outputs = session.run(['logits'], ort_inputs)
        ort_logits = ort_outputs[0]
        
        # Calculate probabilities
        ort_probs = torch.nn.functional.softmax(torch.tensor(ort_logits), dim=-1).numpy()[0]
        
        # Get prediction
        pred_class_idx = np.argmax(ort_probs)
        pred_class = self.class_names[pred_class_idx]
        confidence = float(ort_probs[pred_class_idx])
        
        # Compare with PyTorch model
        with torch.no_grad():
            pt_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pt_outputs = self.model(**pt_inputs)
            pt_logits = pt_outputs.logits.cpu().numpy()
            pt_probs = torch.nn.functional.softmax(torch.tensor(pt_logits), dim=-1).numpy()[0]
            pt_pred_class_idx = np.argmax(pt_probs)
            pt_pred_class = self.class_names[pt_pred_class_idx]
            pt_confidence = float(pt_probs[pt_pred_class_idx])
        
        # Log results
        logger.info(f"ONNX Model Prediction: {pred_class} with confidence {confidence:.4f}")
        logger.info(f"PyTorch Model Prediction: {pt_pred_class} with confidence {pt_confidence:.4f}")
        
        results = {
            "onnx_prediction": {
                "class": pred_class,
                "confidence": confidence,
                "class_idx": int(pred_class_idx),
                "probabilities": ort_probs.tolist()
            },
            "pytorch_prediction": {
                "class": pt_pred_class,
                "confidence": pt_confidence,
                "class_idx": int(pt_pred_class_idx),
                "probabilities": pt_probs.tolist()
            },
            "match": pred_class == pt_pred_class
        }
        
        return results

    def prepare_torchserve_files(self) -> str:
        """
        Prepare files for TorchServe.
        
        Returns:
            Path to the mar file
        """
        logger.info("Preparing files for TorchServe deployment...")
        
        # Create TorchServe directory
        torchserve_dir = os.path.join(self.output_dir, "torchserve")
        os.makedirs(torchserve_dir, exist_ok=True)
        
        # Create model directory
        model_store_dir = os.path.join(torchserve_dir, "model-store")
        os.makedirs(model_store_dir, exist_ok=True)
        
        # Save model and tokenizer for TorchServe
        model_path = os.path.join(torchserve_dir, "model")
        os.makedirs(model_path, exist_ok=True)
        
        # Save model state dict
        model_state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_state_dict_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save config
        self.model.config.save_pretrained(model_path)
        
        # Save label information
        label_config = {
            "id2label": {str(i): label for i, label in enumerate(self.class_names)},
            "label2id": {label: i for i, label in enumerate(self.class_names)}
        }
        
        with open(os.path.join(model_path, "labels.json"), 'w') as f:
            json.dump(label_config, f, indent=2)
        
        # Create handler file
        handler_path = os.path.join(torchserve_dir, "text_classification_handler.py")
        with open(handler_path, 'w') as f:
            f.write("""
import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class TextClassificationHandler(BaseHandler):
    def __init__(self):
        super(TextClassificationHandler, self).__init__()
        self.initialized = False
        
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        
        # Load label mapping
        with open(os.path.join(model_dir, "labels.json"), "r") as f:
            label_config = json.load(f)
        
        self.id2label = label_config["id2label"]
        self.label2id = label_config["label2id"]
        
        # Set max length
        self.max_length = self.model.config.max_position_embeddings if hasattr(self.model.config, "max_position_embeddings") else 512
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.initialized = True
        
    def preprocess(self, data):
        # Get text input
        text_inputs = []
        
        for row in data:
            text = row.get("data")
            if text is None:
                text = row.get("body")
            
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8")
                
            text_inputs.append(text)
            
        # Tokenize
        inputs = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return inputs
    
    def inference(self, inputs):
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs
    
    def postprocess(self, outputs):
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        pred_classes = torch.argmax(probs, dim=-1)
        
        results = []
        for i, pred_class in enumerate(pred_classes):
            pred_class_idx = pred_class.item()
            pred_class_name = self.id2label[str(pred_class_idx)]
            confidence = probs[i][pred_class_idx].item()
            
            # Add all class probabilities
            all_probs = {self.id2label[str(j)]: probs[i][j].item() for j in range(len(self.id2label))}
            
            results.append({
                "prediction": pred_class_name,
                "class_index": pred_class_idx,
                "confidence": confidence,
                "probabilities": all_probs
            })
        
        return results
    
    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)
            
        if data is None:
            return []
            
        inputs = self.preprocess(data)
        outputs = self.inference(inputs)
        results = self.postprocess(outputs)
        
        return results
""")
        
        logger.info(f"Created handler file at {handler_path}")
        
        # Create README with instructions
        readme_path = os.path.join(torchserve_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"""# TorchServe Deployment for {self.model_name}

This directory contains files needed to deploy the model using TorchServe.

## Prerequisites

- Install TorchServe: `pip install torchserve torch-model-archiver torch-workflow-archiver`

## Steps to Deploy

1. Archive the model:

```bash
torch-model-archiver --model-name {self.model_name} \\
                     --version 1.0 \\
                     --serialized-file {os.path.join("model", "pytorch_model.bin")} \\
                     --handler text_classification_handler.py \\
                     --extra-files {os.path.join("model", "config.json")},{os.path.join("model", "vocab.txt")},{os.path.join("model", "labels.json")} \\
                     --export-path model-store
```

2. Start TorchServe:

```bash
torchserve --start --model-store model-store --models {self.model_name}={self.model_name}.mar
```

3. Send a request to the server:

```bash
curl -X POST http://localhost:8080/predictions/{self.model_name} -T input.json
```

where input.json contains:

```json
{{
  "data": "Your text to classify"
}}
```

4. Stop TorchServe when done:

```bash
torchserve --stop
```

## Configuration Files

- `model/`: Contains the model weights, tokenizer, and configuration
- `text_classification_handler.py`: Custom handler for text classification
- `model-store/`: Directory where the model archive (.mar) file will be stored

""")
        
        logger.info(f"Created README with deployment instructions at {readme_path}")
        
        # Create sample input file
        sample_input_path = os.path.join(torchserve_dir, "input.json")
        with open(sample_input_path, 'w') as f:
            f.write('{\n  "data": "This is a sample text for classification"\n}')
        
        logger.info(f"Created sample input file at {sample_input_path}")
        
        # Create shell script to archive and start TorchServe
        script_path = os.path.join(torchserve_dir, "deploy.sh")
        with open(script_path, 'w') as f:
            f.write(f"""#!/bin/bash

# Archive the model
torch-model-archiver --model-name {self.model_name} \\
                     --version 1.0 \\
                     --serialized-file model/pytorch_model.bin \\
                     --handler text_classification_handler.py \\
                     --extra-files model/config.json,model/vocab.txt,model/labels.json \\
                     --export-path model-store

# Start TorchServe
torchserve --start --model-store model-store --models {self.model_name}={self.model_name}.mar

echo "TorchServe started. Test with: curl -X POST http://localhost:8080/predictions/{self.model_name} -T input.json"
""")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created deployment script at {script_path}")
        
        return torchserve_dir

def main():
    """
    Main function to run model deployment.
    """
    parser = argparse.ArgumentParser(description="Deploy transformer model for inference")
    parser.add_argument("--model_dir", type=str, default="output/models/distilbert", 
                        help="Directory containing the model files")
    parser.add_argument("--output_dir", type=str, default="output/deployment",
                        help="Directory to save deployment artifacts")
    parser.add_argument("--test_text", type=str, default="This is a sample text for classification",
                        help="Test text for inference")
    parser.add_argument("--onnx", action="store_true", help="Export model to ONNX format")
    parser.add_argument("--torchserve", action="store_true", help="Prepare files for TorchServe deployment")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ModelDeployer(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Export to ONNX if requested
    if args.onnx:
        onnx_path = deployer.export_to_onnx()
        
        # Test ONNX inference
        test_results = deployer.test_onnx_inference(onnx_path, args.test_text)
        
        # Log results
        logger.info("ONNX Inference Results:")
        logger.info(json.dumps(test_results, indent=2))
        
        # Save results
        results_path = os.path.join(os.path.dirname(onnx_path), "onnx_test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"ONNX test results saved to {results_path}")
    
    # Prepare TorchServe files if requested
    if args.torchserve:
        torchserve_dir = deployer.prepare_torchserve_files()
        logger.info(f"TorchServe deployment files prepared at {torchserve_dir}")
        logger.info("Follow the instructions in the README.md file for deployment.")
    
    # If neither option specified, do both
    if not args.onnx and not args.torchserve:
        onnx_path = deployer.export_to_onnx()
        deployer.test_onnx_inference(onnx_path, args.test_text)
        
        torchserve_dir = deployer.prepare_torchserve_files()
        logger.info(f"TorchServe deployment files prepared at {torchserve_dir}")
    
    logger.info("Model deployment preparation completed.")

if __name__ == "__main__":
    main() 