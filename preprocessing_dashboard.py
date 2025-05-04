import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import logging
import re
import random
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional, Union
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wordcloud
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetAnalyzerDashboard:
    """
    A comprehensive analyzer and visualization dashboard for preprocessed text datasets
    designed for transformer-based classification models.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "analysis_dashboard",
        transformer_model: str = "bert-base-uncased",
        random_seed: int = 42,
        device: str = None
    ):
        """
        Initialize the dataset analyzer dashboard.
        
        Args:
            dataset_dir: Directory containing the preprocessed dataset
            output_dir: Directory to save the analysis results
            transformer_model: Transformer model to use for embeddings
            random_seed: Random seed for reproducibility
            device: Device to use for model inference
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.transformer_model = transformer_model
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.data = {
            'train': None,
            'val': None,
            'test': None
        }
        self.stats = {}
        self.visualizations = {}
        
        # Load NLTK resources
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"Initialized analyzer with model {transformer_model} on {self.device}")
    
    def load_dataset(self) -> None:
        """
        Load the preprocessed dataset from disk.
        """
        logger.info(f"Loading dataset from {self.dataset_dir}...")
        
        # Load each split
        for split in ['train', 'val', 'test']:
            try:
                file_path = os.path.join(self.dataset_dir, f"{split}.csv")
                if os.path.exists(file_path):
                    self.data[split] = pd.read_csv(file_path)
                    logger.info(f"Loaded {split} split with {len(self.data[split])} samples")
                else:
                    logger.warning(f"Split file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {split} split: {e}")
        
        # Load label encoder
        try:
            with open(os.path.join(self.dataset_dir, "label_encoder.pkl"), 'rb') as f:
                self.label_encoder = pickle.load(f)
                logger.info(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
        except Exception as e:
            logger.error(f"Error loading label encoder: {e}")
            self.label_encoder = None
        
        # Load dataset statistics
        try:
            with open(os.path.join(self.dataset_dir, "dataset_stats.json"), 'r', encoding='utf-8') as f:
                self.dataset_stats = json.load(f)
                logger.info("Loaded dataset statistics")
        except Exception as e:
            logger.error(f"Error loading dataset statistics: {e}")
            self.dataset_stats = {}
    
    def compute_basic_statistics(self) -> None:
        """
        Compute basic statistics of the dataset.
        """
        logger.info("Computing basic statistics...")
        
        stats = {}
        
        # Compute statistics for each split
        for split, df in self.data.items():
            if df is not None:
                # Sample counts
                stats[f"{split}_samples"] = len(df)
                
                # Class distribution
                class_counts = df['label'].value_counts().to_dict()
                stats[f"{split}_class_distribution"] = class_counts
                
                # Source distribution
                source_counts = df['source'].value_counts().to_dict()
                stats[f"{split}_source_distribution"] = source_counts
                
                # Text length statistics
                df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
                length_stats = {
                    'mean': df['text_length'].mean(),
                    'median': df['text_length'].median(),
                    'min': df['text_length'].min(),
                    'max': df['text_length'].max(),
                    'std': df['text_length'].std()
                }
                stats[f"{split}_text_length"] = length_stats
        
        # Compute vocabulary statistics
        if self.data['train'] is not None:
            # Count words and vocabulary
            all_words = []
            for text in self.data['train']['text']:
                words = str(text).lower().split()
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            
            # Vocabulary statistics
            vocab_stats = {
                'total_words': len(all_words),
                'unique_words': len(word_counts),
                'top_words': dict(word_counts.most_common(100))
            }
            stats['vocabulary'] = vocab_stats
        
        self.stats = stats
        logger.info("Basic statistics computation complete")
    
    def load_transformer_model(self) -> None:
        """
        Load the transformer model for embeddings.
        """
        logger.info(f"Loading transformer model: {self.transformer_model}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
            self.model = AutoModel.from_pretrained(self.transformer_model)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for the texts using the transformer model.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for inference
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Transformer model not loaded, cannot generate embeddings")
            return None
        
        # Convert to list if it's a pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Filter out None values and ensure all texts are strings
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text is not None and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            elif text is not None:
                # Try to convert non-None values to strings
                try:
                    valid_texts.append(str(text))
                    valid_indices.append(i)
                except:
                    logger.warning(f"Skipping non-string text at index {i}")
        
        if len(valid_texts) == 0:
            logger.error("No valid texts found to generate embeddings")
            return None
        
        # Prepare to store embeddings
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="Generating embeddings"):
            batch_texts = valid_texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use the [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
            
            # Create a full-sized result array with zeros for missing entries
            result_embeddings = np.zeros((len(texts), all_embeddings.shape[1]))
            for i, idx in enumerate(valid_indices):
                result_embeddings[idx] = all_embeddings[i]
            
            return result_embeddings
        
        return None
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        perplexity: int = 30
    ) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Embeddings to reduce
            method: Dimension reduction method ('tsne', 'pca', or 'umap')
            n_components: Number of components to reduce to
            perplexity: Perplexity parameter for t-SNE
            
        Returns:
            Reduced embeddings
        """
        if embeddings is None:
            return None
        
        logger.info(f"Reducing dimensions with {method} to {n_components} components...")
        
        if method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=self.random_seed
            )
        elif method == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=self.random_seed
            )
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_seed
            )
        else:
            logger.error(f"Unknown dimension reduction method: {method}")
            return None
        
        # Apply dimension reduction
        reduced = reducer.fit_transform(embeddings)
        
        logger.info(f"Dimension reduction complete, shape: {reduced.shape}")
        
        return reduced
    
    def visualize_embeddings(
        self,
        reduced_embeddings: np.ndarray,
        labels: List[int],
        sources: List[str],
        method: str,
        split: str
    ) -> None:
        """
        Create a visualization of the reduced embeddings.
        
        Args:
            reduced_embeddings: Reduced embeddings
            labels: Class labels
            sources: Dataset sources
            method: Dimension reduction method
            split: Data split name
        """
        if reduced_embeddings is None or len(reduced_embeddings) == 0:
            logger.error("No embeddings to visualize")
            return
        
        # Get class names
        if self.label_encoder is not None:
            class_names = self.label_encoder.classes_
            label_names = [class_names[label] for label in labels]
        else:
            label_names = [str(label) for label in labels]
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': label_names,
            'source': sources
        })
        
        # Create directory for visualizations
        os.makedirs(os.path.join(self.output_dir, "embeddings"), exist_ok=True)
        
        # Create plot by class
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='label',
            palette='viridis',
            alpha=0.7
        )
        plt.title(f"{split.capitalize()} Set Embeddings by Class ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "embeddings", f"{split}_{method}_by_class.png"))
        plt.close()
        
        # Create plot by source
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='source',
            palette='Set1',
            alpha=0.7
        )
        plt.title(f"{split.capitalize()} Set Embeddings by Source ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "embeddings", f"{split}_{method}_by_source.png"))
        plt.close()
        
        # Create interactive plots with Plotly
        try:
            # By class
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                hover_data=['label', 'source'],
                title=f"{split.capitalize()} Set Embeddings by Class ({method.upper()})"
            )
            fig.update_layout(
                width=1000,
                height=800,
                legend_title_text='Class'
            )
            fig.write_html(os.path.join(self.output_dir, "embeddings", f"{split}_{method}_by_class.html"))
            
            # By source
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='source',
                hover_data=['label', 'source'],
                title=f"{split.capitalize()} Set Embeddings by Source ({method.upper()})"
            )
            fig.update_layout(
                width=1000,
                height=800,
                legend_title_text='Source'
            )
            fig.write_html(os.path.join(self.output_dir, "embeddings", f"{split}_{method}_by_source.html"))
            
            logger.info(f"Created interactive embedding visualizations for {split} set")
        except Exception as e:
            logger.error(f"Error creating interactive plots: {e}")
    
    def analyze_text_similarity(self, split: str = 'train', sample_size: int = 1000) -> None:
        """
        Analyze text similarity within and between classes.
        
        Args:
            split: Data split to analyze
            sample_size: Number of samples to use
        """
        logger.info(f"Analyzing text similarity in {split} set...")
        
        if self.data[split] is None or len(self.data[split]) == 0:
            logger.error(f"No data available for {split} set")
            return
        
        # Sample data if needed
        df = self.data[split]
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=self.random_seed)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['text'])
        if embeddings is None:
            return
        
        # Compute pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Compute average similarity within and between classes
        labels = df['label'].values
        unique_labels = sorted(df['label'].unique())
        
        within_class_similarities = []
        between_class_similarities = []
        
        for i, label_i in enumerate(unique_labels):
            # Get indices for this class
            indices_i = np.where(labels == label_i)[0]
            
            # Within-class similarity
            if len(indices_i) > 1:
                class_similarities = []
                for j in range(len(indices_i)):
                    for k in range(j + 1, len(indices_i)):
                        class_similarities.append(similarity_matrix[indices_i[j], indices_i[k]])
                
                if class_similarities:
                    within_class_similarities.append({
                        'class': label_i,
                        'avg_similarity': np.mean(class_similarities),
                        'std_similarity': np.std(class_similarities)
                    })
            
            # Between-class similarity
            for j, label_j in enumerate(unique_labels[i+1:], start=i+1):
                indices_j = np.where(labels == label_j)[0]
                
                cross_similarities = []
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        cross_similarities.append(similarity_matrix[idx_i, idx_j])
                
                if cross_similarities:
                    between_class_similarities.append({
                        'class_pair': f"{label_i}-{label_j}",
                        'avg_similarity': np.mean(cross_similarities),
                        'std_similarity': np.std(cross_similarities)
                    })
        
        # Store similarity analysis
        self.stats['similarity_analysis'] = {
            'within_class': within_class_similarities,
            'between_class': between_class_similarities
        }
        
        # Create similarity heatmap
        plt.figure(figsize=(14, 12))
        
        # Sort by class
        sorted_indices = np.argsort(labels)
        sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Plot heatmap
        sns.heatmap(sorted_similarity, cmap='viridis')
        plt.title(f"Text Similarity Matrix ({split} set)")
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.join(self.output_dir, "similarity"), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "similarity", f"{split}_similarity_matrix.png"))
        plt.close()
        
        logger.info("Text similarity analysis complete")
    
    def analyze_class_characteristics(self, split: str = 'train') -> None:
        """
        Analyze characteristics of each class including top words and word frequencies.
        
        Args:
            split: Data split to analyze
        """
        logger.info(f"Analyzing class characteristics in {split} set...")
        
        if self.data[split] is None or len(self.data[split]) == 0:
            logger.error(f"No data available for {split} set")
            return
        
        df = self.data[split]
        class_characteristics = {}
        
        # Analyze each class
        for class_label in df['label'].unique():
            # Get texts for this class
            class_texts = df[df['label'] == class_label]['text']
            
            # Count words
            word_counts = Counter()
            for text in class_texts:
                # Tokenize and clean
                words = str(text).lower().split()
                # Remove stopwords
                words = [word for word in words if word not in self.stop_words]
                word_counts.update(words)
            
            # Store class characteristics
            class_characteristics[class_label] = {
                'sample_count': len(class_texts),
                'avg_length': np.mean([len(str(text).split()) for text in class_texts]),
                'std_length': np.std([len(str(text).split()) for text in class_texts]),
                'top_words': dict(word_counts.most_common(30))
            }
            
            # Generate word cloud
            try:
                wc = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    random_state=self.random_seed
                )
                wc.generate_from_frequencies(word_counts)
                
                # Save word cloud
                os.makedirs(os.path.join(self.output_dir, "word_clouds"), exist_ok=True)
                wc.to_file(os.path.join(self.output_dir, "word_clouds", f"{split}_class_{class_label}_wordcloud.png"))
            except Exception as e:
                logger.error(f"Error generating word cloud for class {class_label}: {e}")
        
        # Store analysis results
        self.stats['class_characteristics'] = class_characteristics
        logger.info("Class characteristics analysis complete")
    
    def analyze_data_augmentation(self) -> None:
        """
        Analyze the effect of data augmentation by comparing original and augmented samples.
        """
        logger.info("Analyzing data augmentation...")
        
        # Check if we have source information to identify augmented examples
        if 'train' not in self.data or self.data['train'] is None:
            logger.error("No training data available for augmentation analysis")
            return
        
        df = self.data['train']
        
        # Check if we have augmented samples
        if not 'source' in df.columns or not any('augmented' in str(source).lower() for source in df['source']):
            logger.warning("No augmented samples identified in the data")
            return
        
        # Separate original and augmented samples
        original_df = df[~df['source'].str.contains('augmented', case=False)]
        augmented_df = df[df['source'].str.contains('augmented', case=False)]
        
        logger.info(f"Found {len(original_df)} original samples and {len(augmented_df)} augmented samples")
        
        # Compare text lengths
        original_lengths = [len(str(text).split()) for text in original_df['text']]
        augmented_lengths = [len(str(text).split()) for text in augmented_df['text']]
        
        # Compute statistics
        length_comparison = {
            'original': {
                'mean': np.mean(original_lengths),
                'median': np.median(original_lengths),
                'std': np.std(original_lengths)
            },
            'augmented': {
                'mean': np.mean(augmented_lengths),
                'median': np.median(augmented_lengths),
                'std': np.std(augmented_lengths)
            }
        }
        
        # Compute vocabulary diversity
        original_vocab = set()
        augmented_vocab = set()
        
        for text in original_df['text']:
            words = str(text).lower().split()
            original_vocab.update(words)
        
        for text in augmented_df['text']:
            words = str(text).lower().split()
            augmented_vocab.update(words)
        
        vocab_comparison = {
            'original_vocab_size': len(original_vocab),
            'augmented_vocab_size': len(augmented_vocab),
            'shared_vocab_size': len(original_vocab.intersection(augmented_vocab)),
            'unique_to_original': len(original_vocab - augmented_vocab),
            'unique_to_augmented': len(augmented_vocab - original_vocab)
        }
        
        # Store analysis results
        self.stats['augmentation_analysis'] = {
            'sample_counts': {
                'original': len(original_df),
                'augmented': len(augmented_df)
            },
            'length_comparison': length_comparison,
            'vocabulary_comparison': vocab_comparison
        }
        
        # Visualize length distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(original_lengths, label='Original', alpha=0.6, bins=30)
        sns.histplot(augmented_lengths, label='Augmented', alpha=0.6, bins=30)
        plt.title('Text Length Distribution: Original vs Augmented')
        plt.xlabel('Text Length (words)')
        plt.ylabel('Count')
        plt.legend()
        
        # Save the plot
        os.makedirs(os.path.join(self.output_dir, "augmentation"), exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "augmentation", "length_distribution.png"))
        plt.close()
        
        logger.info("Data augmentation analysis complete")
    
    def analyze_oov_handling(self) -> None:
        """
        Analyze the handling of out-of-vocabulary (OOV) words in the dataset.
        """
        logger.info("Analyzing OOV handling...")
        
        # Check if we have OOV statistics
        oov_stats_path = os.path.join(self.dataset_dir, "oov_analysis", "oov_stats.json")
        
        if not os.path.exists(oov_stats_path):
            logger.warning("OOV statistics not found, skipping analysis")
            return
        
        try:
            with open(oov_stats_path, 'r', encoding='utf-8') as f:
                oov_stats = json.load(f)
        except Exception as e:
            logger.error(f"Error loading OOV statistics: {e}")
            return
        
        # Store OOV statistics
        self.stats['oov_analysis'] = oov_stats
        
        # Visualize OOV ratio
        if 'oov_ratio' in oov_stats:
            plt.figure(figsize=(8, 6))
            plt.pie(
                [oov_stats['oov_ratio'], 1 - oov_stats['oov_ratio']],
                labels=['OOV Words', 'In-Vocabulary Words'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'],
                startangle=90
            )
            plt.title('Proportion of OOV Words in Dataset')
            
            # Save the plot
            os.makedirs(os.path.join(self.output_dir, "oov"), exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, "oov", "oov_ratio.png"))
            plt.close()
        
        # Visualize top OOV words
        if 'top_oov' in oov_stats:
            top_oov = oov_stats['top_oov']
            
            # If it's a list of tuples, convert to dict
            if isinstance(top_oov, list):
                top_oov_dict = {}
                for item in top_oov[:20]:  # Take top 20
                    if isinstance(item, list) and len(item) == 2:
                        top_oov_dict[item[0]] = item[1]
                top_oov = top_oov_dict
            
            # If it's a dict, create bar chart
            if isinstance(top_oov, dict):
                words = list(top_oov.keys())[:20]  # Take top 20
                counts = [top_oov[word] for word in words]
                
                plt.figure(figsize=(12, 8))
                plt.bar(words, counts)
                plt.title('Top 20 OOV Words')
                plt.xlabel('Word')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(os.path.join(self.output_dir, "oov", "top_oov_words.png"))
                plt.close()
        
        logger.info("OOV handling analysis complete")
    
    def create_dataset_summary(self) -> None:
        """
        Create a comprehensive summary of the dataset.
        """
        logger.info("Creating dataset summary...")
        
        # Create summary directory
        summary_dir = os.path.join(self.output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Compile summary statistics
        summary = {
            'dataset_overview': {
                'name': "Multi-Dataset Text Classification",
                'total_samples': sum(self.stats.get(f"{split}_samples", 0) for split in ['train', 'val', 'test']),
                'num_classes': len(self.stats.get('train_class_distribution', {})) if 'train_class_distribution' in self.stats else 0,
                'splits': {
                    'train': self.stats.get('train_samples', 0),
                    'val': self.stats.get('val_samples', 0),
                    'test': self.stats.get('test_samples', 0)
                },
                'sources': list(self.stats.get('train_source_distribution', {}).keys()) if 'train_source_distribution' in self.stats else []
            },
            'text_statistics': {
                'avg_length': self.stats.get('train_text_length', {}).get('mean', 0) if 'train_text_length' in self.stats else 0,
                'vocabulary_size': self.stats.get('vocabulary', {}).get('unique_words', 0) if 'vocabulary' in self.stats else 0
            },
            'class_distribution': self.stats.get('train_class_distribution', {}),
            'preprocessing': {
                'transformer_model': self.transformer_model,
                'oov_handled': 'oov_analysis' in self.stats,
                'augmentation_applied': 'augmentation_analysis' in self.stats
            }
        }
        
        # Save the summary as JSON
        with open(os.path.join(summary_dir, "dataset_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create a summary visualization
        self.create_summary_visualization(summary, summary_dir)
        
        logger.info("Dataset summary created")
    
    def create_summary_visualization(self, summary: Dict[str, Any], output_dir: str) -> None:
        """
        Create a visual summary of the dataset.
        
        Args:
            summary: Summary statistics
            output_dir: Output directory
        """
        # Create multiple plots in a single figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Data Split Distribution',
                'Class Distribution',
                'Source Distribution',
                'Text Length Distribution'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "histogram"}]
            ]
        )
        
        # Plot 1: Data split distribution
        splits = summary['dataset_overview']['splits']
        fig.add_trace(
            go.Pie(
                labels=list(splits.keys()),
                values=list(splits.values()),
                textinfo='percent',
                marker=dict(colors=['#3366cc', '#dc3912', '#ff9900'])
            ),
            row=1, col=1
        )
        
        # Plot 2: Class distribution
        class_dist = summary['class_distribution']
        if class_dist:
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            # If too many classes, group smaller ones
            if len(classes) > 10:
                sorted_idx = np.argsort(counts)[::-1]
                top_classes = [classes[i] for i in sorted_idx[:9]]
                top_counts = [counts[i] for i in sorted_idx[:9]]
                other_count = sum(counts[i] for i in sorted_idx[9:])
                
                classes = top_classes + ['Other']
                counts = top_counts + [other_count]
            
            fig.add_trace(
                go.Bar(
                    x=classes,
                    y=counts,
                    marker_color='#3366cc'
                ),
                row=1, col=2
            )
        
        # Plot 3: Source distribution
        source_dist = self.stats.get('train_source_distribution', {})
        if source_dist:
            fig.add_trace(
                go.Pie(
                    labels=list(source_dist.keys()),
                    values=list(source_dist.values()),
                    textinfo='percent',
                    marker=dict(colors=['#3366cc', '#dc3912', '#ff9900', '#109618'])
                ),
                row=2, col=1
            )
        
        # Plot 4: Text length distribution
        if 'train' in self.data and self.data['train'] is not None:
            text_lengths = [len(str(text).split()) for text in self.data['train']['text']]
            
            fig.add_trace(
                go.Histogram(
                    x=text_lengths,
                    marker_color='#3366cc',
                    nbinsx=30
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Dataset Summary",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Save the figure
        fig.write_html(os.path.join(output_dir, "dataset_summary.html"))
        
        # Also create static version
        fig.write_image(os.path.join(output_dir, "dataset_summary.png"))
    
    def run_full_analysis(self) -> None:
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting full analysis pipeline...")
        
        # Step 1: Load the dataset
        self.load_dataset()
        
        # Step 2: Compute basic statistics
        self.compute_basic_statistics()
        
        # Step 3: Load transformer model
        self.load_transformer_model()
        
        # Step 4: Analyze class characteristics
        self.analyze_class_characteristics(split='train')
        
        # Step 5: Analyze OOV handling
        self.analyze_oov_handling()
        
        # Step 6: Analyze data augmentation
        self.analyze_data_augmentation()
        
        # Step 7: Analyze text similarity (for a subset of training data)
        if self.model is not None:
            self.analyze_text_similarity(split='train', sample_size=500)
        
        # Step 8: Generate and visualize embeddings (for a subset)
        if self.model is not None and 'train' in self.data and self.data['train'] is not None:
            # Sample data to make visualization manageable
            sample_size = min(1000, len(self.data['train']))
            sampled_df = self.data['train'].sample(sample_size, random_state=self.random_seed)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(sampled_df['text'])
            
            if embeddings is not None:
                # Reduce dimensions with t-SNE
                reduced_tsne = self.reduce_dimensions(embeddings, method='tsne')
                if reduced_tsne is not None:
                    self.visualize_embeddings(
                        reduced_tsne,
                        sampled_df['label'].values,
                        sampled_df['source'].values,
                        method='tsne',
                        split='train'
                    )
                
                # Reduce dimensions with PCA
                reduced_pca = self.reduce_dimensions(embeddings, method='pca')
                if reduced_pca is not None:
                    self.visualize_embeddings(
                        reduced_pca,
                        sampled_df['label'].values,
                        sampled_df['source'].values,
                        method='pca',
                        split='train'
                    )
                
                # Reduce dimensions with UMAP if available
                try:
                    reduced_umap = self.reduce_dimensions(embeddings, method='umap')
                    if reduced_umap is not None:
                        self.visualize_embeddings(
                            reduced_umap,
                            sampled_df['label'].values,
                            sampled_df['source'].values,
                            method='umap',
                            split='train'
                        )
                except:
                    logger.warning("UMAP dimension reduction failed, skipping")
        
        # Step 9: Create dataset summary
        self.create_dataset_summary()
        
        # Step 10: Save all statistics
        self.save_analysis_results()
        
        logger.info("Analysis pipeline completed successfully")
    
    def save_analysis_results(self) -> None:
        """
        Save all analysis results to disk.
        """
        # Convert NumPy keys to regular Python types
        def convert_dict_keys(d):
            if not isinstance(d, dict):
                return d
            
            new_dict = {}
            for k, v in d.items():
                # Convert the key if it's a NumPy type
                if isinstance(k, np.integer):  # Use np.integer instead of listing all integer types
                    k = int(k)
                elif isinstance(k, np.floating):  # Use np.floating instead of listing all float types
                    k = float(k)
                
                # Convert the value recursively if it's a dict or list
                if isinstance(v, dict):
                    v = convert_dict_keys(v)
                elif isinstance(v, list):
                    v = [convert_dict_keys(item) if isinstance(item, dict) else item for item in v]
                
                new_dict[k] = v
            
            return new_dict
        
        # First, convert any NumPy keys in dictionaries
        converted_stats = convert_dict_keys(self.stats)
        
        # Then handle any remaining non-serializable values
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):  # Use np.integer for all integer types
                return int(obj)
            elif isinstance(obj, np.floating):  # Use np.floating for all float types
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save statistics as JSON
        with open(os.path.join(self.output_dir, "analysis_statistics.json"), 'w', encoding='utf-8') as f:
            json.dump(converted_stats, f, indent=2, default=convert_to_serializable)
        
        logger.info(f"Analysis results saved to {self.output_dir}")


def main():
    """
    Main function to run the dataset analysis dashboard.
    """
    # Configure parameters
    dataset_dir = "processed_datasets"
    output_dir = "analysis_dashboard"
    transformer_model = "bert-base-uncased"
    
    # Initialize the analyzer
    analyzer = DatasetAnalyzerDashboard(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        transformer_model=transformer_model,
        random_seed=42
    )
    
    # Run the analysis
    analyzer.run_full_analysis()
    
    return 0


if __name__ == "__main__":
    main()