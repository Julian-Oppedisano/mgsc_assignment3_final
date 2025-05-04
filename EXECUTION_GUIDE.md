# Execution Guide for MGSC695 Assignment 3

This guide provides detailed steps for executing the transformer-based text classification project from start to finish.

## Prerequisites

- Python 3.8+ installed
- 8GB+ RAM (16GB recommended)
- GPU recommended but not required (training will be much slower on CPU)
- Good internet connection (for downloading transformer models)

## Setup Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/BTCJULIAN/mgsc695_assignment3.git
   cd mgsc695_assignment3
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Handling Large Model Files

This repository does not include large model files (>100MB) in GitHub due to size restrictions. The large model files are:

- Model safetensors files (*.safetensors)
- PyTorch model files (*.pt)
- ONNX model files (*.onnx)

You have three options for working with models:

### Option 1: Automatic Download

When you run the code, it will automatically download the necessary models from Hugging Face. This requires an internet connection but doesn't require any manual steps.

### Option 2: Manual Download

If you're using the pre-trained models from the submission package:

1. Extract the ZIP file containing model files
2. Place them in the appropriate directories:
   - Main models: `output/models/{model_name}/{model_name}_best.pt`
   - Demo model: `output/demo/model/model.safetensors`

### Option 3: Train from Scratch

Follow the full pipeline execution steps below to train the models from scratch.

## Quick Demo Option

For a quick demonstration without running the full training pipeline:

```bash
python run_demo.py --install_deps
```

This will:
- Install necessary dependencies
- Load a pre-trained model
- Demonstrate model inference
- Show sample explanations and deployment preparation

**Expected output**: Console output showing model predictions, evaluation metrics, and paths to generated explanation files.

## Full Pipeline Execution

### Phase 1: Data Preparation (15-20 minutes)

```bash
python complete_preprocessing_pipeline.py
```

**Expected output**:
- Preprocessed data files in `output/processed_data/`
- Log messages showing processing steps
- Dataset statistics summary

**Files created**:
- `train.csv`, `val.csv`, `test.csv`
- `label_encoder.pkl`
- Various summary statistics files

### Phase 2: Model Training (8-10 hours with GPU, longer with CPU)

```bash
python transformer_trainer.py
```

**Expected output**:
- Training progress for BERT, RoBERTa, and DistilBERT models
- Validation metrics during training
- Final test metrics for each model
- Trained models saved to `output/models/`

**Files created**:
- `output/models/bert/bert-base-uncased_best.pt`
- `output/models/roberta/roberta-base_best.pt`
- `output/models/distilbert/distilbert-base-uncased_best.pt`
- Training history and evaluation reports

### Phase 3: Hyperparameter Tuning (Optional, 3-4 hours)

```bash
python hyperparameter_tuning.py
```

**Expected output**:
- Grid search progress through hyperparameters
- Validation metrics for each configuration
- Visualization plots in `output/hyperparameter_tuning/plots/`

**Files created**:
- Hyperparameter tuning results and plots
- Best hyperparameter configuration report

### Phase 4: Model Evaluation (15-30 minutes)

```bash
python run_phase4.py
```

**Expected output**:
- Detailed evaluation metrics for the DistilBERT model
- Confusion matrix visualization
- Classification report by category
- Attention visualization samples

**Files created**:
- Evaluation reports in `output/evaluation/`
- Confusion matrix plots
- Attention visualization images

### Phase 5: Explainability and Deployment (30-45 minutes)

```bash
python run_phase5.py --model_dir output/models/distilbert --explanation both
```

**Expected output**:
- SHAP and LIME explanations for test examples
- ONNX model export and verification
- TorchServe deployment files preparation

**Files created**:
- Explanations in `output/phase5/explanations/`
- ONNX model in `output/phase5/deployment/`
- TorchServe deployment files

## Examining Results

### Evaluation Results

The model evaluation produces:
- **Classification Report**: Check accuracy, precision, recall, and F1 scores
- **Confusion Matrix**: Visualize prediction errors
- **Attention Weights**: See what the model focuses on

Location: `output/evaluation/`

### Explanation Results

The explainability phase produces:
- **SHAP Explanations**: Global feature importance and local text explanations
- **LIME Explanations**: Word-level importance for predictions

Location: `output/phase5/explanations/`

### Deployment Artifacts

The deployment preparation produces:
- **ONNX Model**: Optimized model for inference
- **TorchServe Files**: Files needed for serving via REST API

Location: `output/phase5/deployment/`

## Troubleshooting Common Issues

### Dependency Issues

**Problem**: Missing or incompatible dependencies
**Solution**: 
```bash
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues

**Problem**: CUDA out of memory errors
**Solution**: Reduce batch size in the configuration files or model parameters

### Model Loading Issues

**Problem**: Model weights not loading correctly
**Solution**: The code includes fallback mechanisms to use the base models. Check logs for warnings about "using base model instead."

### LIME/SHAP Visualization Errors

**Problem**: Errors generating visualizations
**Solution**: Use the `--max_texts` parameter to limit explanations:
```bash
python run_phase5.py --model_dir output/models/distilbert --explanation lime --max_texts 5
```

### ONNX Export Issues

**Problem**: ONNX export fails with operator errors
**Solution**: The code uses opset version 14. If needed, update:
```bash
pip install --upgrade onnx onnxruntime
```

## Expected Time Requirements

- **Complete pipeline**: 9-12 hours (majority in model training)
- **Without training** (using pre-trained models): 1-2 hours
- **Quick demo**: 5-10 minutes

## Additional Information

- Logs are saved in various `.log` files for each component
- Models are saved in PyTorch format with `.pt` extension
- The best performing model is DistilBERT, which balances accuracy and efficiency 