import numpy as np
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import logging
import random
import os
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Tuple, Optional, Union
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTextAugmenter:
    """
    Advanced text augmentation class that implements various techniques including:
    1. Back-translation
    2. Synonym replacement (using WordNet)
    3. EDA (Easy Data Augmentation) techniques
    4. Mixup (text domain)
    """
    
    def __init__(
        self,
        device: str = None,
        cache_dir: str = "model_cache",
        random_seed: int = 42
    ):
        """
        Initialize the advanced text augmenter.
        
        Args:
            device: Device to use for model inference ('cuda' or 'cpu')
            cache_dir: Directory to cache models
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize NLTK resources for synonym replacement
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Define translation model pairs for back-translation
        self.translation_pairs = [
            # Format: (source_to_target_model, target_to_source_model)
            # English -> German -> English
            ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
            # English -> French -> English
            ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
            # English -> Spanish -> English
            ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
            # English -> Russian -> English
            ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en")
        ]
        
        # Initialize translation models
        self.translation_models = {}
        self.translation_tokenizers = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
    
    def load_translation_model(self, model_name: str) -> Tuple[MarianMTModel, MarianTokenizer]:
        """
        Load a translation model and tokenizer with caching.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name in self.translation_models:
            return self.translation_models[model_name], self.translation_tokenizers[model_name]
        
        logger.info(f"Loading translation model: {model_name}")
        
        try:
            # Load the tokenizer
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Load the model
            model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir)
            model.to(self.device)
            model.eval()
            
            # Cache the model and tokenizer
            self.translation_models[model_name] = model
            self.translation_tokenizers[model_name] = tokenizer
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None
    
    def translate(
        self,
        texts: List[str],
        model_name: str,
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[str]:
        """
        Translate a list of texts using the specified model.
        
        Args:
            texts: List of texts to translate
            model_name: Name of the translation model
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            List of translated texts
        """
        model, tokenizer = self.load_translation_model(model_name)
        
        if model is None or tokenizer is None:
            logger.error(f"Failed to load model {model_name}")
            return texts
        
        translated_texts = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            batch_encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Translate
            with torch.no_grad():
                translated = model.generate(**batch_encodings)
            
            # Decode
            batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            translated_texts.extend(batch_translations)
        
        return translated_texts
    
    def back_translate(
        self,
        texts: List[str],
        language_pair_idx: Optional[int] = None,
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[str]:
        """
        Perform back-translation on a list of texts.
        
        Args:
            texts: List of texts to back-translate
            language_pair_idx: Index of the language pair to use (random if None)
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            List of back-translated texts
        """
        # Select a language pair
        if language_pair_idx is None:
            language_pair_idx = random.randint(0, len(self.translation_pairs) - 1)
        
        source_to_target, target_to_source = self.translation_pairs[language_pair_idx]
        
        logger.info(f"Back-translating using: {source_to_target} -> {target_to_source}")
        
        # Translate source -> target
        target_texts = self.translate(texts, source_to_target, batch_size, max_length)
        
        # Translate target -> source
        back_translated_texts = self.translate(target_texts, target_to_source, batch_size, max_length)
        
        return back_translated_texts
    
    def get_wordnet_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Word to find synonyms for
            
        Returns:
            List of synonyms
        """
        synonyms = []
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        
        return synonyms
    
    def synonym_replacement(
        self,
        text: str,
        n_replacements: Optional[int] = None,
        replacement_prob: float = 0.3
    ) -> str:
        """
        Replace words with their synonyms.
        
        Args:
            text: Text to augment
            n_replacements: Number of replacements to make (default: 30% of words)
            replacement_prob: Probability of replacing a word if n_replacements is None
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) <= 3:
            return text
        
        # Determine number of replacements
        if n_replacements is None:
            n_replacements = max(1, int(len(words) * replacement_prob))
        else:
            n_replacements = min(n_replacements, len(words) - 1)
        
        # Select words to replace (avoid first and last word)
        replace_indices = random.sample(range(1, len(words) - 1), n_replacements)
        
        for idx in replace_indices:
            word = words[idx]
            
            # Get synonyms
            synonyms = self.get_wordnet_synonyms(word)
            
            # Replace with a synonym if available
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def random_insertion(
        self,
        text: str,
        n_insertions: Optional[int] = None,
        insertion_ratio: float = 0.1
    ) -> str:
        """
        Randomly insert synonyms into the text.
        
        Args:
            text: Text to augment
            n_insertions: Number of insertions to make (default: 10% of words)
            insertion_ratio: Ratio of words to insert if n_insertions is None
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) <= 3:
            return text
        
        # Determine number of insertions
        if n_insertions is None:
            n_insertions = max(1, int(len(words) * insertion_ratio))
        
        # Perform insertions
        for _ in range(n_insertions):
            # Select a random word to find a synonym for
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            
            # Get synonyms
            synonyms = self.get_wordnet_synonyms(word)
            
            # Insert a synonym if available
            if synonyms:
                synonym = random.choice(synonyms)
                insert_idx = random.randint(0, len(words))
                words.insert(insert_idx, synonym)
        
        return ' '.join(words)
    
    def random_deletion(
        self,
        text: str,
        deletion_prob: float = 0.1
    ) -> str:
        """
        Randomly delete words from the text.
        
        Args:
            text: Text to augment
            deletion_prob: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) <= 3:
            return text
        
        # Make sure we keep at least 3 words
        min_words = max(3, int(len(words) * (1 - deletion_prob)))
        
        # Delete words with a certain probability
        kept_words = []
        
        for word in words:
            if random.random() > deletion_prob or len(kept_words) < min_words - (len(words) - len(kept_words)):
                kept_words.append(word)
        
        # If we deleted all words, return the original text
        if len(kept_words) == 0:
            return text
        
        return ' '.join(kept_words)
    
    def random_swap(
        self,
        text: str,
        n_swaps: Optional[int] = None,
        swap_ratio: float = 0.1
    ) -> str:
        """
        Randomly swap words in the text.
        
        Args:
            text: Text to augment
            n_swaps: Number of swaps to make (default: 10% of words)
            swap_ratio: Ratio of words to swap if n_swaps is None
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) <= 3:
            return text
        
        # Determine number of swaps
        if n_swaps is None:
            n_swaps = max(1, int(len(words) * swap_ratio))
        
        # Perform swaps
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def mixup_text(self, text1: str, text2: str, alpha: float = 0.5) -> str:
        """
        Perform mixup augmentation on two texts.
        This is an adaptation of mixup for text data.
        
        Args:
            text1: First text
            text2: Second text
            alpha: Mixup parameter (0.5 means equal mix)
            
        Returns:
            Mixed text
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Determine the number of words to take from each text
        n_words = max(3, int((len(words1) + len(words2)) * 0.5))
        n_words1 = int(n_words * alpha)
        n_words2 = n_words - n_words1
        
        # Sample words from each text
        if len(words1) > n_words1:
            sampled_words1 = random.sample(words1, n_words1)
        else:
            sampled_words1 = words1
        
        if len(words2) > n_words2:
            sampled_words2 = random.sample(words2, n_words2)
        else:
            sampled_words2 = words2
        
        # Combine and shuffle
        mixed_words = sampled_words1 + sampled_words2
        random.shuffle(mixed_words)
        
        return ' '.join(mixed_words)
    
    def augment_with_various_techniques(
        self,
        texts: List[str],
        labels: List[str],
        techniques: List[str] = None,
        augmentation_factor: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Augment a dataset using various techniques.
        
        Args:
            texts: List of texts to augment
            labels: List of corresponding labels
            techniques: List of techniques to use (default: all)
            augmentation_factor: Number of augmented examples to create per original
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if techniques is None:
            techniques = ['back_translation', 'synonym_replacement', 'random_insertion', 
                         'random_deletion', 'random_swap', 'mixup']
        
        logger.info(f"Augmenting dataset using techniques: {techniques}")
        logger.info(f"Original dataset size: {len(texts)}")
        
        all_augmented_texts = texts.copy()
        all_augmented_labels = labels.copy()
        
        # Track unique texts to avoid duplicates
        unique_texts = set(texts)
        
        # Create augmented examples
        for _ in range(augmentation_factor):
            logger.info(f"Creating augmentation batch {_ + 1}/{augmentation_factor}")
            
            augmented_batch_texts = []
            augmented_batch_labels = []
            
            # Process each original text
            for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="Augmenting")):
                # Choose a random technique
                technique = random.choice(techniques)
                
                augmented_text = text  # Default to original
                
                # Apply the chosen technique
                if technique == 'back_translation' and 'back_translation' in techniques:
                    # Back-translation (process in small batches for memory efficiency)
                    batch_idx = i % 100
                    if batch_idx == 0:
                        batch_texts = [text]
                        batch_augmented = self.back_translate(batch_texts)
                        augmented_text = batch_augmented[0]
                    
                elif technique == 'synonym_replacement' and 'synonym_replacement' in techniques:
                    augmented_text = self.synonym_replacement(text)
                
                elif technique == 'random_insertion' and 'random_insertion' in techniques:
                    augmented_text = self.random_insertion(text)
                
                elif technique == 'random_deletion' and 'random_deletion' in techniques:
                    augmented_text = self.random_deletion(text)
                
                elif technique == 'random_swap' and 'random_swap' in techniques:
                    augmented_text = self.random_swap(text)
                
                elif technique == 'mixup' and 'mixup' in techniques and i > 0:
                    # Mix with a random previous text
                    other_idx = random.randint(0, len(texts) - 1)
                    other_text = texts[other_idx]
                    augmented_text = self.mixup_text(text, other_text)
                
                # Only add if the augmented text is different and not already in the dataset
                if augmented_text != text and augmented_text not in unique_texts:
                    augmented_batch_texts.append(augmented_text)
                    augmented_batch_labels.append(label)
                    unique_texts.add(augmented_text)
            
            # Add the batch of augmented examples
            all_augmented_texts.extend(augmented_batch_texts)
            all_augmented_labels.extend(augmented_batch_labels)
            
            logger.info(f"Added {len(augmented_batch_texts)} new examples in this batch")
        
        logger.info(f"Final augmented dataset size: {len(all_augmented_texts)}")
        
        return all_augmented_texts, all_augmented_labels
    
    def augment_batch_with_back_translation(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> List[str]:
        """
        Augment a batch of texts using back-translation.
        
        Args:
            texts: List of texts to augment
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            List of augmented texts
        """
        # Randomly select a language pair for back-translation
        language_pair_idx = random.randint(0, len(self.translation_pairs) - 1)
        
        # Perform back-translation
        augmented_texts = self.back_translate(
            texts,
            language_pair_idx=language_pair_idx,
            batch_size=batch_size,
            max_length=max_length
        )
        
        return augmented_texts
    
    def augment_dataset_in_batches(
        self,
        texts: List[str],
        labels: List[str],
        technique: str = 'back_translation',
        batch_size: int = 64,
        max_batches: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Augment a dataset in batches, useful for large datasets.
        
        Args:
            texts: List of texts to augment
            labels: List of corresponding labels
            technique: Augmentation technique to use
            batch_size: Size of batches to process
            max_batches: Maximum number of batches to process (None for all)
            
        Returns:
            Tuple of (all_texts, all_labels) including original and augmented data
        """
        all_texts = texts.copy()
        all_labels = labels.copy()
        
        # Determine number of batches
        n_samples = len(texts)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
        
        logger.info(f"Augmenting dataset with {n_batches} batches of size {batch_size}")
        
        # Process each batch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_texts = texts[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{n_batches} with {len(batch_texts)} samples")
            
            # Apply the chosen technique
            if technique == 'back_translation':
                augmented_batch = self.augment_batch_with_back_translation(
                    batch_texts,
                    batch_size=min(8, len(batch_texts))
                )
                
                # Add the augmented batch
                all_texts.extend(augmented_batch)
                all_labels.extend(batch_labels)
            
            elif technique == 'eda':
                # Apply EDA techniques (synonym replacement, random insertion, deletion, swap)
                for text, label in zip(batch_texts, batch_labels):
                    # Apply a random EDA technique
                    eda_technique = random.choice(['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap'])
                    
                    if eda_technique == 'synonym_replacement':
                        augmented_text = self.synonym_replacement(text)
                    elif eda_technique == 'random_insertion':
                        augmented_text = self.random_insertion(text)
                    elif eda_technique == 'random_deletion':
                        augmented_text = self.random_deletion(text)
                    else:  # random_swap
                        augmented_text = self.random_swap(text)
                    
                    # Add the augmented example
                    if augmented_text != text:
                        all_texts.append(augmented_text)
                        all_labels.append(label)
            
            else:
                logger.warning(f"Unknown technique: {technique}, skipping batch")
            
            # Log progress
            logger.info(f"Current dataset size: {len(all_texts)}")
        
        return all_texts, all_labels

# Example usage
def main():
    # Initialize the augmenter
    augmenter = AdvancedTextAugmenter()
    
    # Example texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Climate change is a significant threat to global ecology.",
        "Machine learning algorithms require large amounts of data for training.",
        "Neural networks have revolutionized natural language processing.",
        "The movie was entertaining but lacked character development."
    ]
    sample_labels = ["example"] * len(sample_texts)
    
    # Test back-translation
    logger.info("Testing back-translation...")
    back_translated = augmenter.augment_batch_with_back_translation(sample_texts[:2])
    
    logger.info("Original vs Back-translated:")
    for orig, aug in zip(sample_texts[:2], back_translated):
        logger.info(f"Original: {orig}")
        logger.info(f"Augmented: {aug}")
        logger.info("---")
    
    # Test other augmentation techniques
    logger.info("\nTesting other augmentation techniques:")
    
    text = sample_texts[0]
    logger.info(f"Original: {text}")
    
    # Synonym replacement
    augmented = augmenter.synonym_replacement(text)
    logger.info(f"Synonym replacement: {augmented}")
    
    # Random insertion
    augmented = augmenter.random_insertion(text)
    logger.info(f"Random insertion: {augmented}")
    
    # Random deletion
    augmented = augmenter.random_deletion(text)
    logger.info(f"Random deletion: {augmented}")
    
    # Random swap
    augmented = augmenter.random_swap(text)
    logger.info(f"Random swap: {augmented}")
    
    # Mixup
    augmented = augmenter.mixup_text(sample_texts[0], sample_texts[1])
    logger.info(f"Mixup: {augmented}")
    
    # Test multi-technique augmentation
    logger.info("\nTesting multi-technique augmentation...")
    augmented_texts, augmented_labels = augmenter.augment_with_various_techniques(
        sample_texts,
        sample_labels,
        techniques=['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap'],
        augmentation_factor=2
    )
    
    logger.info(f"Augmented dataset size: {len(augmented_texts)}")
    
    return 0

if __name__ == "__main__":
    main()
