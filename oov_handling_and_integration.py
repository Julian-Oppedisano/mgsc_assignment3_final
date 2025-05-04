import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import nltk
from nltk.corpus import wordnet
from collections import Counter
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
from tqdm import tqdm
import re
import pickle
import time
from datasets import load_dataset

# Import our custom modules
# Assuming they're saved in the same directory
from preprocessing_pipeline import TextPreprocessor
from advanced_augmentation import AdvancedTextAugmenter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OOVHandler:
    """
    Handler for Out-of-Vocabulary (OOV) words, implementing various strategies
    for dealing with words not in the vocabulary.
    """
    
    def __init__(
        self,
        tokenizer,
        max_vocab_size: int = 10000,
        min_freq: int = 5,
        handle_strategy: str = 'subword'
    ):
        """
        Initialize the OOV handler.
        
        Args:
            tokenizer: Tokenizer from the transformer model
            max_vocab_size: Maximum size of custom vocabulary
            min_freq: Minimum frequency for words to include in custom vocabulary
            handle_strategy: Strategy for handling OOV words ('subword', 'unk', 'ignore', 'custom')
        """
        self.tokenizer = tokenizer
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.handle_strategy = handle_strategy
        
        # Initialize word counter and custom vocabulary
        self.word_counter = Counter()
        self.custom_vocab = {}
        self.reverse_vocab = {}
        self.oov_stats = {
            'total_words': 0,
            'oov_words': 0,
            'unique_oov': 0,
            'top_oov': []
        }
        
        # Track character-level n-grams for subword tokenization
        self.char_ngrams = Counter()
    
    def build_vocab_from_texts(self, texts: List[str]) -> None:
        """
        Build a vocabulary from the provided texts.
        
        Args:
            texts: List of texts to analyze
        """
        logger.info("Building vocabulary from texts...")
        
        total_words = 0
        oov_words = 0
        oov_counter = Counter()
        
        # Count all words in the texts
        for text in tqdm(texts, desc="Counting words"):
            # Simple tokenization by splitting on whitespace and removing punctuation
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)
            
            # Update word counter
            self.word_counter.update(words)
            
            # Check OOV words
            for word in words:
                # Check if word is not in the tokenizer's vocabulary
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) > 1 or (len(token_ids) == 1 and token_ids[0] == self.tokenizer.unk_token_id):
                    oov_words += 1
                    oov_counter[word] += 1
                    
                    # Extract character n-grams (for subword tokenization)
                    for n in range(2, 6):  # 2 to 5-grams
                        for i in range(len(word) - n + 1):
                            self.char_ngrams[word[i:i+n]] += 1
        
        # Build custom vocabulary from most common words
        vocab_items = self.word_counter.most_common(self.max_vocab_size)
        vocab_words = [word for word, count in vocab_items if count >= self.min_freq]
        
        # Create mappings
        self.custom_vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.reverse_vocab = {idx: word for word, idx in self.custom_vocab.items()}
        
        # Store OOV statistics
        self.oov_stats = {
            'total_words': total_words,
            'oov_words': oov_words,
            'oov_ratio': oov_words / total_words if total_words > 0 else 0,
            'unique_oov': len(oov_counter),
            'top_oov': oov_counter.most_common(50)
        }
        
        logger.info(f"Vocabulary built with {len(self.custom_vocab)} words")
        logger.info(f"OOV statistics: {self.oov_stats['oov_ratio']:.2%} of words are OOV")
        logger.info(f"Top 10 OOV words: {oov_counter.most_common(10)}")
    
    def handle_oov_words(self, text: str) -> str:
        """
        Apply the selected OOV handling strategy to the text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text with OOV words handled
        """
        if self.handle_strategy == 'ignore':
            # Do nothing
            return text
        
        words = text.split()
        processed_words = []
        
        for word in words:
            # Check if word is OOV
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            is_oov = len(token_ids) > 1 or (len(token_ids) == 1 and token_ids[0] == self.tokenizer.unk_token_id)
            
            if is_oov:
                if self.handle_strategy == 'unk':
                    # Replace with [UNK] token
                    processed_words.append(self.tokenizer.unk_token)
                
                elif self.handle_strategy == 'subword':
                    # Keep the original word (transformer will handle subword tokenization)
                    processed_words.append(word)
                
                elif self.handle_strategy == 'custom':
                    # Use custom vocabulary mapping if available
                    if word.lower() in self.custom_vocab:
                        processed_words.append(word)
                    else:
                        # Try to find the closest word in vocabulary using character overlap
                        closest_word = self._find_closest_word(word)
                        if closest_word:
                            processed_words.append(closest_word)
                        else:
                            processed_words.append(self.tokenizer.unk_token)
            else:
                # Word is in vocabulary, keep it
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def _find_closest_word(self, word: str, min_similarity: float = 0.7) -> Optional[str]:
        """
        Find the closest word in the vocabulary based on character overlap.
        
        Args:
            word: OOV word to find a replacement for
            min_similarity: Minimum similarity threshold
            
        Returns:
            Closest word from vocabulary or None if no word meets the threshold
        """
        if not self.custom_vocab:
            return None
        
        word = word.lower()
        best_similarity = 0
        best_match = None
        
        # Extract character n-grams from the word
        word_ngrams = set()
        for n in range(2, 6):  # 2 to 5-grams
            for i in range(len(word) - n + 1):
                word_ngrams.add(word[i:i+n])
        
        # Compare with words in vocabulary
        for vocab_word in self.custom_vocab:
            # Skip words with large length difference
            if abs(len(vocab_word) - len(word)) > 3:
                continue
            
            # Extract n-grams from vocabulary word
            vocab_ngrams = set()
            for n in range(2, 6):
                for i in range(len(vocab_word) - n + 1):
                    vocab_ngrams.add(vocab_word[i:i+n])
            
            # Calculate Jaccard similarity
            if not word_ngrams or not vocab_ngrams:
                continue
                
            intersection = len(word_ngrams.intersection(vocab_ngrams))
            union = len(word_ngrams.union(vocab_ngrams))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = vocab_word
        
        # Return the best match if it meets the threshold
        if best_similarity >= min_similarity:
            return best_match
        
        return None
    
    def analyze_oov_words(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze OOV words in the texts and return statistics.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing OOV statistics
        """
        logger.info("Analyzing OOV words...")
        
        total_words = 0
        oov_words = 0
        oov_counter = Counter()
        oov_by_position = Counter()
        
        for text in tqdm(texts, desc="Analyzing OOV"):
            words = text.split()
            total_words += len(words)
            
            for i, word in enumerate(words):
                position = "start" if i < len(words) // 4 else "middle" if i < 3 * len(words) // 4 else "end"
                
                # Check if word is OOV
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                is_oov = len(token_ids) > 1 or (len(token_ids) == 1 and token_ids[0] == self.tokenizer.unk_token_id)
                
                if is_oov:
                    oov_words += 1
                    oov_counter[word] += 1
                    oov_by_position[position] += 1
        
        # Compute statistics
        oov_ratio = oov_words / total_words if total_words > 0 else 0
        unique_oov = len(oov_counter)
        
        # Compile results
        results = {
            'total_words': total_words,
            'oov_words': oov_words,
            'oov_ratio': oov_ratio,
            'unique_oov': unique_oov,
            'top_oov': oov_counter.most_common(50),
            'oov_by_position': dict(oov_by_position)
        }
        
        logger.info(f"OOV analysis complete: {oov_ratio:.2%} of words are OOV")
        
        return results
    
    def save_oov_stats(self, output_dir: str = "oov_analysis") -> None:
        """
        Save OOV statistics to disk.
        
        Args:
            output_dir: Directory to save the statistics to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the OOV statistics as JSON
        with open(os.path.join(output_dir, "oov_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(self.oov_stats, f, indent=2)
        
        # Save the top OOV words as CSV
        top_oov_df = pd.DataFrame(self.oov_stats['top_oov'], columns=['word', 'frequency'])
        top_oov_df.to_csv(os.path.join(output_dir, "top_oov_words.csv"), index=False)
        
        # Create a visualization of top OOV words
        plt.figure(figsize=(12, 8))
        sns.barplot(x='frequency', y='word', data=top_oov_df.head(20))
        plt.title('Top 20 OOV Words')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_oov_words.png"))
        plt.close()
        
        logger.info(f"OOV statistics saved to {output_dir}")


class DatasetCombiner:
    """
    Class for combining and preparing multiple datasets for transformer-based text classification.
    """
    
    def __init__(
        self,
        output_dir: str = "combined_dataset",
        random_seed: int = 42,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Initialize the dataset combiner.
        
        Args:
            output_dir: Directory to save the combined dataset
            random_seed: Random seed for reproducibility
            max_samples_per_class: Maximum number of samples per class (for balanced dataset)
        """
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.max_samples_per_class = max_samples_per_class
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataset statistics
        self.dataset_stats = {}
        self.class_mapping = {}
        self.combined_stats = {
            'total_samples': 0,
            'class_distribution': {},
            'source_distribution': {},
            'avg_length': 0
        }
    
    def process_20newsgroups(self, preprocessor) -> Tuple[List[str], List[str], List[str]]:
        # Get texts and labels from the preprocessor
        texts, labels = preprocessor.load_20newsgroups()
        
        # Create a list of sources with the same length as texts
        sources = ["20newsgroups"] * len(texts)
        
        return texts, labels, sources
    
    def process_ag_news(self, preprocessor) -> Tuple[List[str], List[str], List[str]]:
        # Get texts and labels from the preprocessor
        texts, labels = preprocessor.load_ag_news()
        
        # Create a list of sources with the same length as texts
        sources = ["ag_news"] * len(texts)
        
        return texts, labels, sources
    
    def process_imdb(self, preprocessor) -> Tuple[List[str], List[str], List[str]]:
        # Get texts and labels from the preprocessor
        texts, labels = preprocessor.load_imdb()
        
        # Create a list of sources with the same length as texts
        sources = ["imdb"] * len(texts)
        
        return texts, labels, sources
    
    def combine_datasets(
        self,
        preprocessor: TextPreprocessor,
        use_20newsgroups: bool = True,
        use_ag_news: bool = True,
        use_imdb: bool = True,
        balance_classes: bool = True,
        create_unified_labels: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Combine multiple datasets with options for balancing and label unification.
        
        Args:
            preprocessor: Text preprocessor
            use_20newsgroups: Whether to include the 20 Newsgroups dataset
            use_ag_news: Whether to include the AG News dataset
            use_imdb: Whether to include the IMDB dataset
            balance_classes: Whether to balance classes
            create_unified_labels: Whether to create unified labels across datasets
            
        Returns:
            Tuple of (combined_texts, combined_labels, sources)
        """
        all_texts = []
        all_labels = []
        all_sources = []
        
        # Process each selected dataset
        if use_20newsgroups:
            texts, labels, sources = self.process_20newsgroups(preprocessor)
            all_texts.extend(texts)
            all_labels.extend([f"20ng_{label}" for label in labels] if create_unified_labels else labels)
            all_sources.extend(["20newsgroups"] * len(texts))
            
            # Store dataset statistics
            self.dataset_stats['20newsgroups'] = {
                'samples': len(texts),
                'unique_labels': len(set(labels)),
                'avg_length': np.mean([len(text.split()) for text in texts])
            }
        
        if use_ag_news:
            texts, labels, sources = self.process_ag_news(preprocessor)
            all_texts.extend(texts)
            all_labels.extend([f"ag_{label}" for label in labels] if create_unified_labels else labels)
            all_sources.extend(["ag_news"] * len(texts))
            
            # Store dataset statistics
            self.dataset_stats['ag_news'] = {
                'samples': len(texts),
                'unique_labels': len(set(labels)),
                'avg_length': np.mean([len(text.split()) for text in texts])
            }
        
        if use_imdb:
            texts, labels, sources = self.process_imdb(preprocessor)
            all_texts.extend(texts)
            all_labels.extend([f"imdb_{label}" for label in labels] if create_unified_labels else labels)
            all_sources.extend(["imdb"] * len(texts))
            
            # Store dataset statistics
            self.dataset_stats['imdb'] = {
                'samples': len(texts),
                'unique_labels': len(set(labels)),
                'avg_length': np.mean([len(text.split()) for text in texts])
            }
        
        # Create unified label mapping
        unique_labels = sorted(set(all_labels))
        self.class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Balance classes if requested
        if balance_classes:
            df = pd.DataFrame({
                'text': all_texts,
                'label': all_labels,
                'source': all_sources
            })
            
            # Group by label and count samples
            label_counts = df['label'].value_counts()
            
            # Determine number of samples per class
            if self.max_samples_per_class is None:
                samples_per_class = min(label_counts)
            else:
                samples_per_class = min(self.max_samples_per_class, min(label_counts))
            
            logger.info(f"Balancing dataset with {samples_per_class} samples per class")
            
            # Sample from each class
            balanced_df = pd.DataFrame()
            for label in label_counts.index:
                class_df = df[df['label'] == label]
                
                # If we have more samples than needed, downsample
                if len(class_df) > samples_per_class:
                    downsampled = class_df.sample(samples_per_class, random_state=self.random_seed)
                    balanced_df = pd.concat([balanced_df, downsampled])
                else:
                    # Otherwise keep all samples from this class
                    balanced_df = pd.concat([balanced_df, class_df])
            
            # Shuffle the balanced dataframe
            balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            
            # Extract balanced data
            all_texts = balanced_df['text'].tolist()
            all_labels = balanced_df['label'].tolist()
            all_sources = balanced_df['source'].tolist()
        
        # Update combined statistics
        self.combined_stats = {
            'total_samples': len(all_texts),
            'class_distribution': dict(pd.Series(all_labels).value_counts()),
            'source_distribution': dict(pd.Series(all_sources).value_counts()),
            'avg_length': np.mean([len(text.split()) for text in all_texts]),
            'class_mapping': self.class_mapping
        }
        
        logger.info(f"Combined dataset has {len(all_texts)} samples across {len(self.class_mapping)} classes")
        
        return all_texts, all_labels, all_sources
    
    def apply_oov_handling(
        self,
        texts: List[str],
        labels: List[str],
        sources: List[str],
        oov_handler: OOVHandler
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Apply OOV handling to the texts.
        
        Args:
            texts: List of texts
            labels: List of labels
            sources: List of sources
            oov_handler: OOV handler
            
        Returns:
            Tuple of (processed_texts, labels, sources)
        """
        logger.info(f"Applying OOV handling with strategy: {oov_handler.handle_strategy}")
        
        # Build vocabulary first
        oov_handler.build_vocab_from_texts(texts)
        
        # Process each text
        processed_texts = []
        for text in tqdm(texts, desc="Handling OOV words"):
            processed_text = oov_handler.handle_oov_words(text)
            processed_texts.append(processed_text)
        
        # Analyze OOV words after processing
        oov_analysis = oov_handler.analyze_oov_words(processed_texts)
        
        # Save OOV statistics
        oov_handler.save_oov_stats(os.path.join(self.output_dir, "oov_analysis"))
        
        return processed_texts, labels, sources
    
    def apply_augmentation(
        self,
        texts: List[str],
        labels: List[str],
        sources: List[str],
        augmenter: AdvancedTextAugmenter,
        augmentation_factor: int = 1,
        techniques: List[str] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Apply data augmentation to the texts.
        
        Args:
            texts: List of texts
            labels: List of labels
            sources: List of sources
            augmenter: Text augmenter
            augmentation_factor: Number of augmented examples to create per original
            techniques: List of augmentation techniques to use
            
        Returns:
            Tuple of (augmented_texts, augmented_labels, augmented_sources)
        """
        if techniques is None:
            techniques = ['back_translation', 'synonym_replacement', 'random_insertion', 
                         'random_deletion', 'random_swap']
        
        logger.info(f"Applying data augmentation with techniques: {techniques}")
        
        # Apply augmentation
        augmented_texts, augmented_labels = augmenter.augment_with_various_techniques(
            texts=texts,
            labels=labels,
            techniques=techniques,
            augmentation_factor=augmentation_factor
        )
        
        # Update sources for augmented data
        augmented_sources = sources.copy()
        for _ in range(len(augmented_texts) - len(sources)):
            idx = min(_ % len(sources), len(sources) - 1)
            augmented_sources.append(f"{sources[idx]}_augmented")
        
        return augmented_texts, augmented_labels, augmented_sources
    
    def split_dataset(
        self,
        texts: List[str],
        labels: List[str],
        sources: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True
    ) -> Dict[str, Union[List, np.ndarray]]:
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            texts: List of texts
            labels: List of labels
            sources: List of sources
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            stratify: Whether to stratify the splits by label
            
        Returns:
            Dictionary containing the split data
        """
        logger.info("Splitting dataset...")
        
        # Create label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Convert to numpy arrays
        texts_array = np.array(texts)
        sources_array = np.array(sources)
        
        # First split: train+val vs test
        stratify_param = encoded_labels if stratify else None
        X_train_val, X_test, y_train_val, y_test, src_train_val, src_test = train_test_split(
            texts_array, encoded_labels, sources_array,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_param
        )
        
        # Second split: train vs val
        adjusted_val_size = val_size / (1 - test_size)
        stratify_param = y_train_val if stratify else None
        X_train, X_val, y_train, y_val, src_train, src_val = train_test_split(
            X_train_val, y_train_val, src_train_val,
            test_size=adjusted_val_size,
            random_state=self.random_seed,
            stratify=stratify_param
        )
        
        # Log split statistics
        logger.info(f"Data split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
        
        # Prepare split data
        split_data = {
            'train_texts': X_train.tolist(),
            'train_labels': y_train,
            'train_sources': src_train.tolist(),
            'val_texts': X_val.tolist(),
            'val_labels': y_val,
            'val_sources': src_val.tolist(),
            'test_texts': X_test.tolist(),
            'test_labels': y_test,
            'test_sources': src_test.tolist(),
            'label_encoder': label_encoder,
            'class_names': label_encoder.classes_.tolist()
        }
        
        return split_data
    
    def save_dataset(self, split_data: Dict[str, Any]) -> None:
        """
        Save the processed and split dataset.
        
        Args:
            split_data: Dictionary containing the split data
        """
        logger.info(f"Saving dataset to {self.output_dir}...")
        
        # Save the label encoder as a pickle file
        with open(os.path.join(self.output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(split_data['label_encoder'], f)

        # Create pandas DataFrames for each split and save as CSV
        # Training split
        train_df = pd.DataFrame({
            'text': split_data['train_texts'],
            'label': split_data['train_labels'],
            'source': split_data['train_sources']
        })
        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        
        # Validation split
        val_df = pd.DataFrame({
            'text': split_data['val_texts'],
            'label': split_data['val_labels'],
            'source': split_data['val_sources']
        })
        val_df.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)
        
        # Test split
        test_df = pd.DataFrame({
            'text': split_data['test_texts'],
            'label': split_data['test_labels'],
            'source': split_data['test_sources']
        })
        test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        # For JSON, only save the class names
        split_data['label_classes'] = split_data['label_encoder'].classes_.tolist()
        del split_data['label_encoder']

        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert the data to be JSON serializable
        serializable_data = make_json_serializable(split_data)

        # Save to JSON
        with open(os.path.join(self.output_dir, "dataset.json"), "w") as f:
            json.dump(serializable_data, f, indent=2)
        
        # Save dataset statistics (required by preprocessing_dashboard.py)
        dataset_stats = {
            'total_samples': {
                'train': len(split_data['train_texts']),
                'val': len(split_data['val_texts']),
                'test': len(split_data['test_texts'])
            },
            'class_distribution': self.combined_stats.get('class_distribution', {}),
            'source_distribution': self.combined_stats.get('source_distribution', {}),
            'avg_length': self.combined_stats.get('avg_length', 0),
            'class_mapping': self.class_mapping
        }
        
        # Save dataset stats
        with open(os.path.join(self.output_dir, "dataset_stats.json"), "w") as f:
            json.dump(make_json_serializable(dataset_stats), f, indent=2)
        
        logger.info("Dataset saved successfully")
    
    def visualize_dataset(self) -> None:
        """
        Create visualizations of the dataset statistics.
        """
        logger.info("Creating dataset visualizations...")
        
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Class distribution
        if self.combined_stats['class_distribution']:
            plt.figure(figsize=(14, 8))
            
            # Sort by class count
            classes = list(self.combined_stats['class_distribution'].keys())
            counts = list(self.combined_stats['class_distribution'].values())
            sorted_idx = np.argsort(counts)[::-1]
            
            # Plot
            sns.barplot(
                x=[classes[i] for i in sorted_idx], 
                y=[counts[i] for i in sorted_idx]
            )
            plt.title('Class Distribution')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "class_distribution.png"))
            plt.close()
        
        # Source distribution
        if self.combined_stats['source_distribution']:
            plt.figure(figsize=(10, 6))
            
            # Sort by source count
            sources = list(self.combined_stats['source_distribution'].keys())
            counts = list(self.combined_stats['source_distribution'].values())
            
            # Plot
            sns.barplot(x=sources, y=counts)
            plt.title('Source Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "source_distribution.png"))
            plt.close()
        
        logger.info(f"Visualizations saved to {vis_dir}")


def main():
    """
    Main function to demonstrate the integrated preprocessing pipeline.
    """
    # Initialize the text preprocessor
    preprocessor = TextPreprocessor(
        transformer_model="bert-base-uncased",
        max_length=256,
        do_stemming=True,
        do_lemmatization=False,
        remove_stopwords=True,
        remove_punctuation=True,
        lowercase=True,
        random_seed=42
    )
    
    # Initialize the dataset combiner
    dataset_combiner = DatasetCombiner(
        output_dir="processed_datasets",
        random_seed=42,
        max_samples_per_class=5000  # Limit to 5000 samples per class for balanced dataset
    )
    
    # Combine datasets
    texts, labels, sources = dataset_combiner.combine_datasets(
        preprocessor=preprocessor,
        use_20newsgroups=True,
        use_ag_news=True,
        use_imdb=True,
        balance_classes=True,
        create_unified_labels=True
    )
    
    # Initialize the OOV handler
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    oov_handler = OOVHandler(
        tokenizer=tokenizer,
        max_vocab_size=30000,
        min_freq=5,
        handle_strategy='subword'
    )
    
    # Apply OOV handling
    processed_texts, processed_labels, processed_sources = dataset_combiner.apply_oov_handling(
        texts=texts,
        labels=labels,
        sources=sources,
        oov_handler=oov_handler
    )
    
    # Initialize the text augmenter
    augmenter = AdvancedTextAugmenter(
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir="model_cache",
        random_seed=42
    )
    
    # Apply data augmentation
    augmented_texts, augmented_labels, augmented_sources = dataset_combiner.apply_augmentation(
        texts=processed_texts,
        labels=processed_labels,
        sources=processed_sources,
        augmenter=augmenter,
        augmentation_factor=1,  # Create 1 augmented example per original
        techniques=['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap']
    )
    
    # Split the dataset
    split_data = dataset_combiner.split_dataset(
        texts=augmented_texts,
        labels=augmented_labels,
        sources=augmented_sources,
        test_size=0.2,
        val_size=0.1,
        stratify=True
    )
    
    # Save the dataset
    dataset_combiner.save_dataset(split_data)
    
    # Create visualizations
    dataset_combiner.visualize_dataset()
    
    logger.info("Preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()