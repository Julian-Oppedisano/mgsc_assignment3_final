
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
