from typing import Any, List
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

import ngram

EMBEDDING_SIZE = 50
WINDOW_SIZE = 5
MIN_COUNT = ngram.UNKNOWN_THRESHOLD
PAD_TOKEN = "<PAD>"

class WordEmbeddingManager:
    _model = None
    def __init__(self, model_path:str = None):
        self._model = None
        if model_path is not None:
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        if os.path.exists(model_path):
            self._model = Word2Vec.load(model_path)
            print("Model loaded successfully from", model_path)
        else:
            print("Model path does not exist.")

    def train_model(self, corpus: list, size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=4):
        self._model = Word2Vec(corpus, vector_size=size, window=window, min_count=min_count, workers=workers)
        print("Model trained successfully.")
    
    def save_model(self, model_path):
        self._model.save(model_path)
        print("Model saved successfully to: ", model_path)
    
    def get_embedding(self, word:str):
        if self._model is None:
            raise Exception("Model is not loaded.")
        if word in self._model.wv.key_to_index:
            vector_index = self._model.wv.key_to_index[word]
            return self._model.wv.get_vector(vector_index)
        else:  # Handles both <PAD> and unknown words
            return np.zeros(self._model.vector_size)

class SentenceEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset for creating embeddings from sentences.

    Attributes:
        sentences (List[List[str]]): A list of tokenized sentences.
        embedding_manager (Any): An embedding manager instance with a method 'get_embedding'.
        max_sequence_length (int): The maximum length of a sequence.
    """
    def __init__(self, sentences: list, embedding_manager: WordEmbeddingManager, sequence_length: int):
        self.sentences = sentences
        self.embedding_manager = embedding_manager
        self.sequence_length = sequence_length
    
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
        embeddings = np.zeros((self.sequence_length, self.embedding_manager._model.vector_size))
        for i, word in enumerate(sentence):
            embeddings[i] = self.embedding_manager.get_embedding(word)
        return torch.tensor(embeddings, dtype=torch.float32)
    
def create_embedding_dataloader(sentences: List[List[str]], embedding_manager: Any, seq_length: int, batch_size: int) -> DataLoader:
    """
    Creates a DataLoader for sentences with embeddings.

    Args:
        sentences (List[List[str]]): The tokenized sentences.
        embedding_manager (Any): An instance of the embedding manager to retrieve word embeddings.
        seq_length (int): The maximum sequence length of sentences.
        batch_size (int): The number of items per batch.

    Returns:
        DataLoader: A PyTorch DataLoader object ready to be used in a training or evaluation loop.
    """
    dataset = SentenceEmbeddingDataset(sentences, embedding_manager, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
