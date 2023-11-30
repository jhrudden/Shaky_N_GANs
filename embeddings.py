from typing import Any, List, Literal
from gensim.models import Word2Vec
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

import ngram

EMBEDDING_SIZE = 50
WINDOW_SIZE = 5
MIN_COUNT = ngram.UNKNOWN_THRESHOLD
PAD_TOKEN = "<PAD>"

class WordEmbeddingManager:
    """
    A class to handle interfacing with a Word2Vec model either loaded from disk or trained from a corpus.

    Attributes:
        _model (gensim.models.Word2Vec): The Word2Vec model.
    """
    _model = None
    def __init__(self, model_path:str = None):
        self._model = None
        if model_path is not None:
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        """
        Loads a Word2Vec model from disk.

        Args:
            model_path (str): The path to the model file.
        """
        if os.path.exists(model_path):
            self._model = Word2Vec.load(model_path)
            print("Model loaded successfully from", model_path)
        else:
            print("Model path does not exist.")

    def train_model(self, corpus: list, size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=4):
        """
        Trains a Word2Vec model from a corpus of sentences. The model is stored in the _model attribute.

        Args:
            corpus (list): The corpus to train the model on.
            size (int, optional): The size of the word embeddings. Defaults to EMBEDDING_SIZE.
            window (int, optional): The window size for the Word2Vec model. Defaults to WINDOW_SIZE.
            min_count (int, optional): The minimum count for a word to be included in the vocabulary. Defaults to MIN_COUNT.
            workers (int, optional): The number of workers to use for training. Defaults to 4.
        """
        self._model = Word2Vec(corpus, vector_size=size, window=window, min_count=min_count, workers=workers)
        print("Model trained successfully.")
    
    def save_model(self, model_path):
        """
        Saves the Word2Vec model to disk.
        
        Args:
            model_path (str): The path to save the model to.
        """
        self._model.save(model_path)
        print("Model saved successfully to: ", model_path)
    
    def get_embedding(self, word:str):
        """
        Returns the Word2Vec embedding for a given word. If the word is not in the vocabulary, a zero vector is returned.

        Args:
            word (str): The word to retrieve the embedding for.
        Return (np.ndarray): The embedding for the word.
        """
        if self._model is None:
            raise Exception("Model is not loaded.")
        if word in self._model.wv.key_to_index:
            vector_index = self._model.wv.key_to_index[word]
            return self._model.wv.get_vector(vector_index)
        else:  # Handles both <PAD> and unknown words
            return np.zeros(self._model.vector_size)
    
    def one_hot_encode(self, word:str):
        """
        Returns the one-hot encoding for a given word based on its vocab index in the Word2Vec model.
        Zero index is reserved for padding token.

        Args:
            word (str): The word to encode.
        
        Returns:
            torch.Tensor: A one-hot encoded tensor representing the word.
        """
        if self._model is None:
            raise Exception("Model is not loaded.")
        if word == PAD_TOKEN:
            return torch.nn.functional.one_hot(torch.tensor(0), num_classes=len(self._model.wv.key_to_index) + 1).float()
        if word in self._model.wv.key_to_index:
            vector_index = self._model.wv.key_to_index[word] + 1  # +1 to accommodate padding token offset
            return torch.nn.functional.one_hot(torch.tensor(vector_index), num_classes=len(self._model.wv.key_to_index) + 1).float()
        else:
            raise ValueError("Word not in vocabulary.")

    def decode_one_hot(self, one_hot_vector: torch.Tensor) -> str:
        """
        Returns the word corresponding to a given one-hot encoded tensor.

        Args:
            one_hot_vector (torch.Tensor): The one-hot encoded tensor representing a word.

        Returns:
            str: The word corresponding to the one-hot encoding.

        Raises:
            Exception: If the Word2Vec model is not loaded.
            ValueError: If the one-hot encoding does not correspond to any word in the vocabulary.
        """
        if self._model is None:
            raise Exception("Model is not loaded.")
        
        if one_hot_vector.size()[0] != len(self._model.wv.key_to_index) + 1:
            raise ValueError("Input tensor is not a valid one-hot encoding.")

        # Check if input tensor is one-hot encoded
        if torch.sum(one_hot_vector) != 1:
            raise ValueError("Input tensor is not a valid one-hot encoding.")

        # find index of 1 in one-hot vector
        index = torch.argmax(one_hot_vector).item()

        if index == 0:
            return PAD_TOKEN

        # Map index back to word, considering the offset for padding token
        word = self._model.wv.index_to_key[index - 1]  # -1 to accommodate padding token offset
        return word

    def index_to_word(self, index: int) -> str:
        """
        Returns the word corresponding to a given index in the Word2Vec model's vocabulary.

        Args:
            index (int): The index of the word to retrieve.

        Returns:
            str: The word corresponding to the index.

        Raises:
            Exception: If the Word2Vec model is not loaded.
            ValueError: If the index does not correspond to any word in the vocabulary.
        """
        if self._model is None:
            raise Exception("Model is not loaded.")
        if index == 0:
            return PAD_TOKEN
        if index - 1 < len(self._model.wv.index_to_key):
            return self._model.wv.index_to_key[index - 1]
        else:
            raise ValueError("Index does not correspond to any word in the vocabulary.")

class SentenceEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset for preparing sentences with embeddings or encodings for training.

    Attributes:
        sentences (List[List[str]]): A list of tokenized sentences.
        embedding_manager (Any): An embedding manager instance with a method 'get_embedding'.
        max_sequence_length (int): The maximum length of a sequence.
        encoding_method (Literal["word_embedding", "one_hot"]): The encoding method to use. Defaults to "word_embedding".
        verbose (bool): Whether or not to print debug information.

    """
    def __init__(self, sentences: list, embedding_manager: WordEmbeddingManager, sequence_length: int, encoding_method: Literal["word_embedding", "one_hot"] = "word_embedding", verbose: bool=False):
        self.sentences = sentences
        self.embedding_manager = embedding_manager
        self.sequence_length = sequence_length
        self.encoding_method = encoding_method
        self.verbose = verbose
    
    def __len__(self):
        """Returns the number of sentences in the dataset."""
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Retrieves the embeddings for the sentence at the specified index in the dataset.

        Args:
            idx (int): The index of the sentence to retrieve.

        Returns:
            torch.Tensor: A tensor of embeddings for the sentence.
        """
        sentence = self.sentences[idx]
        sentence = sentence[:self.sequence_length] + [PAD_TOKEN] * (self.sequence_length - len(sentence))
        if self.verbose:
            print(f'Encoding sentence: {sentence}, with encoding method: {self.encoding_method}')
        if self.encoding_method == "word_embedding":
            embeddings = np.zeros((self.sequence_length, self.embedding_manager._model.vector_size))
            for i, word in enumerate(sentence):
                embeddings[i] = self.embedding_manager.get_embedding(word)
            return torch.tensor(embeddings, dtype=torch.float32)
        elif self.encoding_method == "one_hot":
            # get one hot encodings for each word in the sentence
            one_hot_encodings = [self.embedding_manager.one_hot_encode(word) for word in sentence]
            # stack the one hot encodings into a tensor
            return torch.stack(one_hot_encodings, dim=0)
        else:
            raise ValueError(f"Invalid encoding method: {self.encoding_method}.")
    
def create_embedding_dataloader(sentences: List[List[str]], embedding_manager: Any, seq_length: int, batch_size: int, encoding_method: Literal["word_embedding", "one_hot"] = "word_embedding", verbose: bool=False) -> DataLoader:
    """
    Creates a DataLoader for sentences with embeddings.

    Args:
        sentences (List[List[str]]): The tokenized sentences.
        embedding_manager (Any): An instance of the embedding manager to retrieve word embeddings.
        seq_length (int): The maximum sequence length of sentences.
        batch_size (int): The number of items per batch.
        encoding_method (Literal["word_embedding", "one_hot"], optional): The encoding method to use. Defaults to "word_embedding".
        verbose (bool, optional): Whether or not to print debug information. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader object ready to be used in a training or evaluation loop.
    """
    dataset = SentenceEmbeddingDataset(sentences, embedding_manager, seq_length, encoding_method=encoding_method, verbose=verbose)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

