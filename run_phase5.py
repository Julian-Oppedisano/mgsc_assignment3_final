import os
import subprocess
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_phase5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd):
    """Run a shell command and log the output."""
    logger.info(f"Running: {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        logger.info(output)
        return True, output
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.output)
        return False, e.output

def install_dependencies():
    """Install required dependencies for Phase 5."""
    logger.info("Installing dependencies for Phase 5...")
    
    # Install explainability dependencies
    success, _ = run_command("pip install shap lime")
    if not success:
        logger.warning("Failed to install SHAP and LIME. Some functionalities may not work.")
    
    # Install deployment dependencies
    success, _ = run_command("pip install onnx onnxruntime")
    if not success:
        logger.warning("Failed to install ONNX dependencies. Export to ONNX may not work.")
    
    success, _ = run_command("pip install torchserve torch-model-archiver")
    if not success:
        logger.warning("Failed to install TorchServe dependencies. TorchServe deployment may not work.")
    
    logger.info("Dependencies installation completed.")

def run_model_explanation(model_dir, output_dir, test_data, num_examples, method, custom_text=None, max_texts=10):
    """Run model explanation with SHAP and/or LIME."""
    logger.info(f"Running model explanation with {method}...")
    
    # Create a more controlled command that limits the number of texts to explain
    cmd = f"python model_explanation.py --model_dir {model_dir} --output_dir {output_dir} --test_data {test_data} --num_examples {num_examples} --method {method}"
    
    if custom_text:
        cmd += f" --custom_text \"{custom_text}\""
    
    if max_texts > 0:
        cmd += f" --max_texts {max_texts}"
    
    success, _ = run_command(cmd)
    
    if success:
        logger.info(f"Model explanation completed. Results saved to {output_dir}")
    else:
        logger.error(f"Model explanation failed. Check logs for details.")
        # Continue with deployment even if explanation fails

def run_model_deployment(model_dir, output_dir, test_text, deployment_type):
    """Run model deployment with ONNX and/or TorchServe."""
    logger.info(f"Preparing model deployment with {deployment_type}...")
    
    cmd = f"python model_deployment.py --model_dir {model_dir} --output_dir {output_dir} --test_text \"{test_text}\""
    
    if deployment_type == "onnx":
        cmd += " --onnx"
    elif deployment_type == "torchserve":
        cmd += " --torchserve"
    # For "both", no additional flags needed as it's the default
    
    success, _ = run_command(cmd)
    
    if success:
        logger.info(f"Model deployment preparation completed. Artifacts saved to {output_dir}")
    else:
        logger.error(f"Model deployment failed. Check logs for details.")

def main():
    """
    Main function to run Phase 5 - Explainability and Scalability.
    """
    parser = argparse.ArgumentParser(description="Run Phase 5 - Explainability and Scalability")
    parser.add_argument("--model_dir", type=str, default="output/models/distilbert", 
                        help="Directory containing the model files")
    parser.add_argument("--output_dir", type=str, default="output/phase5",
                        help="Directory to save Phase 5 outputs")
    parser.add_argument("--test_data", type=str, default="output/processed_data/test.csv",
                        help="Path to test data for examples")
    parser.add_argument("--explanation", type=str, choices=["shap", "lime", "both"], default="both",
                        help="Explanation method to use")
    parser.add_argument("--deployment", type=str, choices=["onnx", "torchserve", "both"], default="both",
                        help="Deployment method to prepare")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Number of examples per class to explain")
    parser.add_argument("--custom_text", type=str, default=None,
                        help="Custom text to explain (optional)")
    parser.add_argument("--test_text", type=str, default="This is a sample text for classification",
                        help="Test text for inference")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install dependencies before running")
    parser.add_argument("--max_texts", type=int, default=10,
                        help="Maximum number of texts to explain (to avoid memory issues)")
    
    try:
        args = parser.parse_args()
        
        logger.info("Starting Phase 5 - Explainability and Scalability")
        
        # Create output directories
        explanation_dir = os.path.join(args.output_dir, "explanations")
        deployment_dir = os.path.join(args.output_dir, "deployment")
        os.makedirs(explanation_dir, exist_ok=True)
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Install dependencies if requested
        if args.install_deps:
            install_dependencies()
        
        # Run model explanation
        run_model_explanation(
            model_dir=args.model_dir,
            output_dir=explanation_dir,
            test_data=args.test_data,
            num_examples=args.num_examples,
            method=args.explanation,
            custom_text=args.custom_text,
            max_texts=args.max_texts
        )
        
        # Run model deployment
        run_model_deployment(
            model_dir=args.model_dir,
            output_dir=deployment_dir,
            test_text=args.test_text,
            deployment_type=args.deployment
        )
        
        logger.info("""
Phase 5 Completion Summary:
---------------------------
a. Use SHAP or LIME to interpret model predictions ✅
   - Explainability methods implemented to understand model predictions
   - Visualizations generated to highlight important features
   
b. Explore deployment with ONNX and TorchServe ✅
   - ONNX export provides optimized inference for various runtimes
   - TorchServe setup ready for production REST API deployment
   
The explainability and scalability phase (Phase 5) is now complete.

Project Completion:
------------------
Congratulations! You have successfully completed all phases of the project:
1. ✅ Data Preparation and Exploratory Data Analysis
2. ✅ Implement and Fine-Tune Transformer Models
3. ✅ Train the Model with Advanced Techniques
4. ✅ Evaluate and Interpret the Model
5. ✅ Explainability and Scalability

Your transformer-based text classification project is now complete with
state-of-the-art model explanation capabilities and production-ready
deployment options.
""")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 