import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import logging
import random
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import os
import json
import pickle
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for transformer-based text classification
    that handles multiple datasets and advanced preprocessing techniques.
    """
    
    def __init__(
        self,
        transformer_model: str = "bert-base-uncased",
        max_length: int = 512,
        do_stemming: bool = True,
        do_lemmatization: bool = False,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the text preprocessor with configurable preprocessing options.
        
        Args:
            transformer_model: The pre-trained transformer model name for tokenization
            max_length: Maximum sequence length for the transformer
            do_stemming: Whether to apply stemming
            do_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert text to lowercase
            random_seed: Random seed for reproducibility
        """
        self.transformer_model = transformer_model
        self.max_length = max_length
        self.do_stemming = do_stemming
        self.do_lemmatization = do_lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Initialize preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize the tokenizer for the transformer model
        logger.info(f"Loading tokenizer for {transformer_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Statistics collection
        self.vocab_stats = {}
        self.class_distribution = {}
        self.oov_words = Counter()
        self.sequence_length_stats = {}
        
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning based on the configured options.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase if enabled
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation if enabled
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if enabled
        if self.do_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization if enabled
        if self.do_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Rejoin tokens into a single string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def load_20newsgroups(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the 20 Newsgroups dataset.
        
        Returns:
            Tuple containing lists of preprocessed texts and their corresponding labels
        """
        logger.info("Loading 20 Newsgroups dataset...")
        
        # Load the 20 Newsgroups dataset
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        texts = newsgroups.data
        labels = [newsgroups.target_names[label] for label in newsgroups.target]
        
        # Preprocess the texts
        logger.info("Preprocessing 20 Newsgroups texts...")
        preprocessed_texts = []
        for text in tqdm(texts, desc="Preprocessing 20 Newsgroups"):
            preprocessed_texts.append(self.clean_text(text))
        
        # Log dataset statistics
        self._log_dataset_stats("20_newsgroups", preprocessed_texts, labels)
        
        return preprocessed_texts, labels
    
    def load_ag_news(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the AG News dataset.
        
        Returns:
            Tuple containing lists of preprocessed texts and their corresponding labels
        """
        logger.info("Loading AG News dataset...")
        
        # Load the AG News dataset
        ag_news = load_dataset("ag_news")
        
        # Extract texts and labels
        texts = ag_news["train"]["text"] + ag_news["test"]["text"]
        labels = ag_news["train"]["label"] + ag_news["test"]["label"]
        
        # Convert numerical labels to string labels
        label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        string_labels = [label_map[label] for label in labels]
        
        # Preprocess the texts
        logger.info("Preprocessing AG News texts...")
        preprocessed_texts = []
        for text in tqdm(texts, desc="Preprocessing AG News"):
            preprocessed_texts.append(self.clean_text(text))
        
        # Log dataset statistics
        self._log_dataset_stats("ag_news", preprocessed_texts, string_labels)
        
        return preprocessed_texts, string_labels
    
    def load_imdb(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the IMDb Reviews dataset.
        
        Returns:
            Tuple containing lists of preprocessed texts and their corresponding labels
        """
        logger.info("Loading IMDb Reviews dataset...")
        
        # Load the IMDb dataset
        imdb = load_dataset("imdb")
        
        # Extract texts and labels
        train_texts = imdb["train"]["text"]
        train_labels = imdb["train"]["label"]
        test_texts = imdb["test"]["text"]
        test_labels = imdb["test"]["label"]
        
        texts = train_texts + test_texts
        labels = train_labels + test_labels
        
        # Convert numerical labels to string labels
        label_map = {
            0: "Negative",
            1: "Positive"
        }
        string_labels = [label_map[label] for label in labels]
        
        # Preprocess the texts
        logger.info("Preprocessing IMDb texts...")
        preprocessed_texts = []
        for text in tqdm(texts, desc="Preprocessing IMDb"):
            preprocessed_texts.append(self.clean_text(text))
        
        # Log dataset statistics
        self._log_dataset_stats("imdb", preprocessed_texts, string_labels)
        
        return preprocessed_texts, string_labels
    
    def _log_dataset_stats(self, dataset_name: str, texts: List[str], labels: List[str]) -> None:
        """
        Log statistics about the dataset for analysis.
        
        Args:
            dataset_name: Name of the dataset
            texts: List of preprocessed texts
            labels: List of labels
        """
        # Text length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        # Update class distribution
        label_counts = Counter(labels)
        
        # Log findings
        logger.info(f"{dataset_name} statistics:")
        logger.info(f"  Number of samples: {len(texts)}")
        logger.info(f"  Number of classes: {len(label_counts)}")
        logger.info(f"  Average text length: {np.mean(text_lengths):.2f} words")
        logger.info(f"  Median text length: {np.median(text_lengths):.2f} words")
        logger.info(f"  Min text length: {min(text_lengths)} words")
        logger.info(f"  Max text length: {max(text_lengths)} words")
        logger.info(f"  Class distribution: {dict(label_counts)}")
        
        # Store statistics
        self.sequence_length_stats[dataset_name] = {
            "mean": float(np.mean(text_lengths)),
            "median": float(np.median(text_lengths)),
            "min": min(text_lengths),
            "max": max(text_lengths),
            "std": float(np.std(text_lengths))
        }
        
        self.class_distribution[dataset_name] = dict(label_counts)
    
    def combine_datasets(
        self,
        use_20newsgroups: bool = True,
        use_ag_news: bool = True,
        use_imdb: bool = True,
        balance_classes: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load and combine multiple datasets with options for class balancing.
        
        Args:
            use_20newsgroups: Whether to include the 20 Newsgroups dataset
            use_ag_news: Whether to include the AG News dataset
            use_imdb: Whether to include the IMDb Reviews dataset
            balance_classes: Whether to balance classes by downsampling
            
        Returns:
            Tuple containing lists of combined texts, labels, and dataset sources
        """
        all_texts = []
        all_labels = []
        all_sources = []
        
        # Load selected datasets
        if use_20newsgroups:
            texts, labels = self.load_20newsgroups()
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_sources.extend(["20newsgroups"] * len(texts))
        
        if use_ag_news:
            texts, labels = self.load_ag_news()
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_sources.extend(["ag_news"] * len(texts))
        
        if use_imdb:
            texts, labels = self.load_imdb()
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_sources.extend(["imdb"] * len(texts))
        
        # Balance classes if requested
        if balance_classes:
            logger.info("Balancing classes by downsampling...")
            df = pd.DataFrame({
                'text': all_texts,
                'label': all_labels,
                'source': all_sources
            })
            
            # Get counts of each class
            class_counts = df['label'].value_counts()
            min_class_count = class_counts.min()
            
            # Downsample each class
            balanced_df = pd.DataFrame()
            for label in class_counts.index:
                class_df = df[df['label'] == label]
                downsampled = class_df.sample(min_class_count, random_state=self.random_seed)
                balanced_df = pd.concat([balanced_df, downsampled])
            
            # Shuffle the balanced dataframe
            balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            
            all_texts = balanced_df['text'].tolist()
            all_labels = balanced_df['label'].tolist()
            all_sources = balanced_df['source'].tolist()
            
            logger.info(f"After balancing: {len(all_texts)} samples")
        
        return all_texts, all_labels, all_sources
    
    def augment_text(self, text: str, label: str, augmentation_type: str = 'random') -> str:
        """
        Apply data augmentation to the given text.
        
        Args:
            text: Input text to augment
            label: Label of the text (for potential class-specific augmentation)
            augmentation_type: Type of augmentation to apply (random, synonym, etc.)
            
        Returns:
            Augmented text
        """
        tokens = text.split()
        
        if len(tokens) < 3:
            return text
        
        # Choose augmentation technique
        if augmentation_type == 'random' or augmentation_type not in ['synonym', 'insertion', 'deletion', 'swap']:
            augmentation_type = random.choice(['synonym', 'insertion', 'deletion', 'swap'])
        
        if augmentation_type == 'synonym':
            # Synonym replacement (simplified version)
            # In a real scenario, you'd use WordNet or another resource for synonyms
            synonyms = {
                'good': ['great', 'excellent', 'amazing', 'wonderful'],
                'bad': ['poor', 'terrible', 'awful', 'horrible'],
                'happy': ['joyful', 'pleased', 'delighted', 'content'],
                'sad': ['unhappy', 'depressed', 'melancholy', 'gloomy'],
                'important': ['crucial', 'significant', 'essential', 'vital']
            }
            
            for i, token in enumerate(tokens):
                if token in synonyms and random.random() < 0.3:  # 30% chance of replacement
                    tokens[i] = random.choice(synonyms[token])
        
        elif augmentation_type == 'insertion':
            # Random word insertion
            for _ in range(max(1, int(len(tokens) * 0.1))):  # Insert up to 10% more words
                insert_position = random.randint(0, len(tokens))
                # Insert a random word from the original text
                insert_word = random.choice(tokens)
                tokens.insert(insert_position, insert_word)
        
        elif augmentation_type == 'deletion':
            # Random word deletion
            if len(tokens) > 5:  # Only delete if we have enough words
                for _ in range(max(1, int(len(tokens) * 0.1))):  # Delete up to 10% of words
                    if len(tokens) > 3:  # Ensure we keep at least 3 words
                        delete_position = random.randint(0, len(tokens) - 1)
                        tokens.pop(delete_position)
        
        elif augmentation_type == 'swap':
            # Random word swap
            for _ in range(max(1, int(len(tokens) * 0.1))):  # Swap up to 10% of adjacent words
                if len(tokens) > 2:  # Need at least 2 words to swap
                    swap_position = random.randint(0, len(tokens) - 2)
                    tokens[swap_position], tokens[swap_position + 1] = tokens[swap_position + 1], tokens[swap_position]
        
        return ' '.join(tokens)
    
    def create_augmented_data(
        self,
        texts: List[str],
        labels: List[str],
        sources: List[str],
        augmentation_factor: int = 1,
        augmentation_techniques: List[str] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Create augmented data by applying multiple augmentation techniques.
        
        Args:
            texts: List of original texts
            labels: List of original labels
            sources: List of dataset sources
            augmentation_factor: Number of augmented copies per original text
            augmentation_techniques: List of augmentation techniques to use
            
        Returns:
            Tuple containing augmented texts, labels, and sources
        """
        if augmentation_factor <= 0:
            return texts, labels, sources
        
        if augmentation_techniques is None:
            augmentation_techniques = ['random']
        
        logger.info(f"Augmenting data with factor {augmentation_factor}...")
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        augmented_sources = sources.copy()
        
        for i, (text, label, source) in enumerate(tqdm(zip(texts, labels, sources), desc="Augmenting data", total=len(texts))):
            for _ in range(augmentation_factor):
                technique = random.choice(augmentation_techniques)
                augmented_text = self.augment_text(text, label, technique)
                
                # Only add the augmented text if it's different from the original
                if augmented_text != text:
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
                    augmented_sources.append(f"{source}_augmented")
        
        # Shuffle the augmented data
        combined = list(zip(augmented_texts, augmented_labels, augmented_sources))
        random.shuffle(combined)
        augmented_texts, augmented_labels, augmented_sources = zip(*combined)
        
        logger.info(f"After augmentation: {len(augmented_texts)} samples")
        
        return list(augmented_texts), list(augmented_labels), list(augmented_sources)
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode string labels to numeric labels using LabelEncoder.
        
        Args:
            labels: List of string labels
            
        Returns:
            Numpy array of encoded numeric labels
        """
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        logger.info(f"Encoded {len(self.label_encoder.classes_)} classes: {list(self.label_encoder.classes_)}")
        
        return encoded_labels
    
    def split_data(
        self,
        texts: List[str],
        labels: List[str],
        sources: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True
    ) -> Dict[str, Union[List, np.ndarray]]:
        """
        Split data into training, validation, and test sets with optional stratification.
        
        Args:
            texts: List of texts
            labels: List of labels
            sources: List of dataset sources
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            stratify: Whether to stratify the splits by label
            
        Returns:
            Dictionary containing the split data
        """
        # Encode labels
        encoded_labels = self.encode_labels(labels)
        
        # Convert to numpy arrays for easier handling
        texts_array = np.array(texts)
        sources_array = np.array(sources)
        
        # First split: training+validation vs test
        stratify_param = encoded_labels if stratify else None
        X_train_val, X_test, y_train_val, y_test, src_train_val, src_test = train_test_split(
            texts_array, encoded_labels, sources_array,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_param
        )
        
        # Second split: training vs validation
        # Adjust validation size to account for the test split
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
        
        # Return as dictionary
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
            'label_encoder': self.label_encoder
        }
        
        return split_data
    
    def tokenize_text(
        self,
        texts: List[str],
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts using the transformer tokenizer.
        
        Args:
            texts: List of texts to tokenize
            padding: Padding strategy
            truncation: Whether to truncate to max_length
            return_tensors: Type of tensors to return
            
        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def analyze_tokenization(self, texts: List[str]) -> None:
        """
        Analyze the tokenization of texts and log statistics about token distribution.
        
        Args:
            texts: List of texts to analyze
        """
        logger.info("Analyzing tokenization...")
        
        # Tokenize a sample of texts
        sample_size = min(1000, len(texts))
        sample_texts = random.sample(texts, sample_size)
        
        # Get token counts
        token_counts = []
        oov_counts = []
        
        for text in tqdm(sample_texts, desc="Analyzing tokens"):
            # Get tokens without special tokens
            tokens = self.tokenizer.tokenize(text)
            token_counts.append(len(tokens))
            
            # Check for OOV tokens (tokenized with ##)
            oov_tokens = [token for token in tokens if '##' in token or token in self.tokenizer.special_tokens_map.values()]
            oov_counts.append(len(oov_tokens))
            
            # Update OOV counter
            for token in oov_tokens:
                self.oov_words[token] += 1
        
        # Log statistics
        logger.info(f"Token count statistics (sample of {sample_size} texts):")
        logger.info(f"  Average token count: {np.mean(token_counts):.2f}")
        logger.info(f"  Median token count: {np.median(token_counts):.2f}")
        logger.info(f"  Min token count: {min(token_counts)}")
        logger.info(f"  Max token count: {max(token_counts)}")
        
        logger.info(f"OOV token statistics:")
        logger.info(f"  Average OOV token count: {np.mean(oov_counts):.2f}")
        logger.info(f"  Total unique OOV tokens: {len(self.oov_words)}")
        logger.info(f"  Top 10 OOV tokens: {self.oov_words.most_common(10)}")
        
        # Update vocab statistics
        self.vocab_stats = {
            "avg_tokens": float(np.mean(token_counts)),
            "median_tokens": float(np.median(token_counts)),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_oov": float(np.mean(oov_counts)),
            "top_oov": dict(self.oov_words.most_common(20))
        }
    
    def save_preprocessed_data(
        self,
        split_data: Dict[str, Union[List, np.ndarray]],
        output_dir: str = "preprocessed_data"
    ) -> None:
        """
        Save preprocessed data and metadata to disk.
        
        Args:
            split_data: Dictionary containing the split data
            output_dir: Directory to save the data to
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save split data
        for split in ['train', 'val', 'test']:
            texts = split_data[f'{split}_texts']
            labels = split_data[f'{split}_labels']
            sources = split_data[f'{split}_sources']
            
            # Save texts and metadata
            with open(os.path.join(output_dir, f"{split}_texts.txt"), 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(f"{text}\n")
            
            # Save labels
            np.save(os.path.join(output_dir, f"{split}_labels.npy"), labels)
            
            # Save sources
            with open(os.path.join(output_dir, f"{split}_sources.txt"), 'w', encoding='utf-8') as f:
                for source in sources:
                    f.write(f"{source}\n")
        
        # Save label encoder
        with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(split_data['label_encoder'], f)
        
        # Save preprocessing statistics
        preprocessing_stats = {
            "vocab_stats": self.vocab_stats,
            "class_distribution": self.class_distribution,
            "sequence_length_stats": self.sequence_length_stats,
            "top_oov_words": dict(self.oov_words.most_common(100)),
            "label_classes": split_data['label_encoder'].classes_.tolist(),
            "preprocessing_config": {
                "transformer_model": self.transformer_model,
                "max_length": self.max_length,
                "do_stemming": self.do_stemming,
                "do_lemmatization": self.do_lemmatization,
                "remove_stopwords": self.remove_stopwords,
                "remove_punctuation": self.remove_punctuation,
                "lowercase": self.lowercase
            }
        }
        
        with open(os.path.join(output_dir, "preprocessing_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(preprocessing_stats, f, indent=2)
        
        logger.info(f"Saved preprocessed data to {output_dir}")
    
    def create_transformer_dataset(self, split_data: Dict[str, Union[List, np.ndarray]]) -> Dict[str, Dataset]:
        """
        Create PyTorch datasets for training, validation, and testing.
        
        Args:
            split_data: Dictionary containing the split data
            
        Returns:
            Dictionary containing PyTorch datasets for each split
        """
        class TransformerDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'label': torch.tensor(label, dtype=torch.long)
                }
        
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            datasets[split] = TransformerDataset(
                split_data[f'{split}_texts'],
                split_data[f'{split}_labels'],
                self.tokenizer,
                self.max_length
            )
        
        logger.info(f"Created transformer datasets: {[(split, len(dataset)) for split, dataset in datasets.items()]}")
        
        return datasets

# Example usage of the TextPreprocessor class
def main():
    # Initialize the preprocessor with custom options
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
    
    # Load and combine datasets
    texts, labels, sources = preprocessor.combine_datasets(
        use_20newsgroups=True,
        use_ag_news=True,
        use_imdb=True,
        balance_classes=True
    )
    
    # Apply data augmentation
    texts, labels, sources = preprocessor.create_augmented_data(
        texts=texts,
        labels=labels,
        sources=sources,
        augmentation_factor=1,  # Create 1 augmented version for each text
        augmentation_techniques=['synonym', 'insertion', 'deletion', 'swap']
    )
    
    # Analyze the tokenization
    preprocessor.analyze_tokenization(texts)
    
    # Split the data
    split_data = preprocessor.split_data(
        texts=texts,
        labels=labels,
        sources=sources,
        test_size=0.2,
        val_size=0.1,
        stratify=True
    )
    
    # Save the preprocessed data
    preprocessor.save_preprocessed_data(split_data, output_dir="preprocessed_data")
    
    # Create PyTorch datasets
    datasets = preprocessor.create_transformer_dataset(split_data)
    
    logger.info("Preprocessing complete!")
    
    return split_data, datasets

if __name__ == "__main__":
    main()