# MGSC695 Assignment 3: Transformer-Based Text Classification

This project implements a comprehensive text classification system using transformer models, covering data preprocessing, model training, evaluation, model explanation, and deployment.

## Project Overview

The project demonstrates the complete lifecycle of a text classification system using state-of-the-art transformer models:

1. **Data Preparation**: Process and clean text data, explore dataset characteristics
2. **Transformer Models**: Implement and compare BERT, RoBERTa, and DistilBERT
3. **Advanced Training**: Apply optimization techniques like AdamW, learning rate scheduling, mixed precision
4. **Model Evaluation**: Evaluate with comprehensive metrics and visualizations
5. **Explainability & Deployment**: Make models interpretable and ready for production

## Project Structure

```
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
│
├── Preprocessing Pipeline
│   ├── preprocessing_pipeline.py     # Core data preprocessing functionality
│   ├── complete_preprocessing_pipeline.py # End-to-end preprocessing
│   └── preprocessing_dashboard.py    # Visualization of preprocessing steps
│
├── Model Implementation
│   ├── transformer_trainer.py        # Implementation of transformer models
│   ├── hyperparameter_tuning.py      # Hyperparameter optimization
│   └── experiment_runner.py          # Run experiments with different settings
│
├── Model Evaluation
│   ├── model_evaluation.py           # Evaluation metrics and analysis
│   ├── run_phase4.py                 # Run the evaluation pipeline
│   └── phase4_demo.py                # Simplified demo of evaluation
│
├── Explainability & Deployment
│   ├── model_explanation.py          # SHAP and LIME explanations
│   ├── model_deployment.py           # ONNX and TorchServe deployment
│   ├── run_phase5.py                 # Run explainability and deployment
│   └── run_demo.py                   # Simplified demo of full pipeline
│
└── output/                           # Stored models, results and artifacts
    ├── processed_data/               # Preprocessed datasets
    ├── models/                       # Trained models
    ├── evaluation/                   # Evaluation results
    ├── explanations/                 # Model explanations
    └── deployment/                   # Deployment artifacts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mgsc695_assignment3.git
cd mgsc695_assignment3

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Important Note About Model Files

Due to GitHub's file size limitations, model files (*.pt, *.safetensors, *.bin) are not included in this repository. To use the project:

### Option 1: Download Pre-trained Models Automatically

The code will automatically download the transformer models from Hugging Face during the first run.

### Option 2: Train Models from Scratch

Run the complete training pipeline following the instructions in this README to generate models locally.

### Option 3: Get Models from Project Submission

For course evaluation, all model files are included in the submitted ZIP file but excluded from GitHub. To use them, place the models in the appropriate directories (described in EXECUTION_GUIDE.md).

## Running the Project

### Quick Demo

To run a simplified demo that demonstrates the project's capabilities without full training:

```bash
python run_demo.py --install_deps
```

### Full Pipeline Execution

To run the complete project pipeline from start to finish:

#### 1. Data Preprocessing

```bash
python complete_preprocessing_pipeline.py
```

This prepares the data for model training, including tokenization, cleaning, and train/val/test splitting.

#### 2. Model Training

```bash
python transformer_trainer.py
```

This trains and compares BERT, RoBERTa, and DistilBERT models, saving the trained models to `output/models/`.

#### 3. Optional: Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

This fine-tunes the best-performing model (DistilBERT) with various hyperparameter configurations.

#### 4. Model Evaluation

```bash
python run_phase4.py
```

This evaluates the model's performance with metrics, confusion matrices, and attention visualization.

#### 5. Explainability and Deployment

```bash
python run_phase5.py --model_dir output/models/distilbert --explanation both
```

This generates SHAP and LIME explanations for model predictions and prepares the model for deployment.

## Phase Details

### Phase 1: Data Preparation

- **Data Cleaning**: Remove noise, handle missing values
- **Tokenization**: Prepare text for transformer models
- **EDA**: Explore class distribution, text length, vocabulary
- **Data Augmentation**: Enhance training with additional examples

### Phase 2: Transformer Model Implementation

- **Models Compared**: BERT, RoBERTa, DistilBERT
- **Custom Components**: Enhanced attention mechanisms, classification heads
- **Performance**: DistilBERT achieved best balance of performance and efficiency

### Phase 3: Advanced Training Techniques

- **Optimization**: AdamW optimizer with weight decay
- **Learning Rate**: Scheduled learning rate with warmup
- **Mixed Precision**: Faster training with FP16/FP32 precision
- **Training Management**: Early stopping, checkpointing

### Phase 4: Evaluation and Visualization

- **Metrics**: Accuracy, precision, recall, F1 score
- **Confusion Matrix**: Visualize model's error patterns
- **Attention Visualization**: Understand what the model focuses on

### Phase 5: Explainability and Deployment

- **SHAP**: Global feature importance and local explanations
- **LIME**: Word-level explanations for predictions
- **ONNX**: Model export for optimized inference
- **TorchServe**: Deployment-ready model serving

## Troubleshooting

If you encounter issues with Phase 5 (especially LIME explanations or ONNX export):

1. **LIME Visualization Errors**: Use the `--max_texts` parameter to limit the number of explanations:
   ```bash
   python run_phase5.py --model_dir output/models/distilbert --explanation lime --max_texts 5
   ```

2. **ONNX Export Errors**: The code uses ONNX opset version 14 to support newer operators. If you encounter errors, try updating onnxruntime:
   ```bash
   pip install --upgrade onnx onnxruntime
   ```

3. **Model Loading Errors**: The project includes fallback mechanisms to handle model compatibility issues. If you see warnings about mismatched state dictionaries, the code will still run with the base model.

## Results

The project demonstrates:

- DistilBERT outperformed BERT and RoBERTa on our dataset
- Advantages of advanced training techniques for transformer models
- Importance of model explainability for understanding predictions
- Production-ready deployment options for transformer models

## Dependencies

Key dependencies include:
- PyTorch
- Transformers (Hugging Face)
- SHAP and LIME for explainability
- ONNX and TorchServe for deployment
- Scikit-learn for evaluation metrics
- Matplotlib and Seaborn for visualization
