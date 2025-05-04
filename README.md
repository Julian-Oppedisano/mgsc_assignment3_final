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

The project demonstrates the performance of three transformer models (BERT, RoBERTa, and DistilBERT) on text classification tasks. Below are detailed results based on comprehensive evaluations.

### Model Performance Comparison

The models were evaluated using standard classification metrics:

- **DistilBERT**: 
  - Accuracy: 84.83%
  - Weighted F1 Score: 84.84%
  - Weighted Precision: 85.03%
  - Weighted Recall: 84.83%

- **BERT**:
  - Accuracy: 76.67%
  - Weighted F1 Score: 76.52%
  - Weighted Precision: 76.67%
  - Weighted Recall: 76.67%

- **RoBERTa**:
  - Accuracy: 76.25%
  - Weighted F1 Score: 76.01%
  - Weighted Precision: 76.23%
  - Weighted Recall: 76.25%

### Key Findings

1. **Model Efficiency and Performance**:
   - DistilBERT outperformed both BERT and RoBERTa by a significant margin (~8% higher accuracy)
   - DistilBERT achieved this superior performance despite having fewer parameters
   - The smaller model size makes DistilBERT more suitable for deployment scenarios with limited resources

2. **Per-Class Performance Analysis**:
   - Best performing classes (DistilBERT):
     - Class 22: 95.96% F1 score (highest)
     - Class 10: 94.79% F1 score
     - Class 23: 92.68% F1 score
   - Most challenging classes:
     - Class 19: 66.83% F1 score (lowest)
     - Class 3: 76.13% F1 score
     - Class 2: 77.36% F1 score

3. **Precision vs. Recall Trade-offs**:
   - DistilBERT showed balanced precision and recall across most classes
   - Notable exceptions:
     - Class 7: Higher recall (88.37%) than precision (69.60%)
     - Class 14: Higher precision (92.35%) than recall (83.03%)

4. **Model Comparison Insights**:
   - DistilBERT consistently outperformed the other models across all metrics
   - BERT slightly outperformed RoBERTa, but the difference was minimal
   - All models struggled with similar classes, indicating inherent classification challenges in those categories

### Hyperparameter Tuning Results

The hyperparameter tuning for DistilBERT revealed:

- Optimal learning rate: 5e-5
- Optimal batch size: 16
- Dropout rate: 0.1
- Attention dropout: 0.1
- Weight decay: 0.01

These parameters provided the best balance between performance and computational efficiency, while helping to prevent overfitting during training.

### Model Interpretability Insights

The model explainability analysis provided valuable insights into the decision-making process:

1. **Text Features**:
   - Certain domain-specific keywords strongly influenced classification decisions
   - Punctuation and formatting sometimes played a significant role in classification
   - The models effectively learned to ignore common stopwords

2. **Attention Patterns**:
   - The attention visualization showed that the model focuses on discriminative terms specific to each class
   - The self-attention mechanism effectively captured relationships between relevant terms in long texts
   - Models demonstrated the ability to focus on key phrases even in the presence of noisy text

3. **Classification Errors**:
   - Misclassifications often occurred with closely related categories
   - Ambiguous texts containing multiple topics were more likely to be misclassified
   - Short texts with limited context posed challenges for accurate classification

### Deployment Performance

When deployed using ONNX and TorchServe:

- Model size reduction: DistilBERT model reduced from 255MB to approximately 134MB after ONNX conversion
- Inference speed: Approximately 30ms per request on standard hardware (CPU-only)
- Processing capacity: Able to handle 30+ requests per second on a single CPU core


## Dependencies

Key dependencies include:
- PyTorch
- Transformers (Hugging Face)
- SHAP and LIME for explainability
- ONNX and TorchServe for deployment
- Scikit-learn for evaluation metrics
- Matplotlib and Seaborn for visualization
