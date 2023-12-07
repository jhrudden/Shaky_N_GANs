from typing import Any, List, Literal
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import json

import ngram

WINDOW_SIZE = 5
MIN_COUNT = ngram.UNKNOWN_THRESHOLD
PAD_TOKEN = "<PAD>"

class EncodingManager:
    """
    A class to handle creating and managing encodings for words.

    Attributes:
        word_to_index (dict): A dictionary mapping words to their corresponding indices.
    """
    word_to_index = None
    def __init__(self, dict_path:str = None):
        self.word_to_index = None
        if dict_path is not None:
            # must be a json file
            assert dict_path.endswith('.json')
            self._load_encoding_dict(dict_path)
    
    def _load_encoding_dict(self, dict_path:str):
        """
        Loads a word to index pairs from disk.

        Args:
            model_path (str): The path to the model file.
        """
        if os.path.exists(dict_path):
            # load json of word_to_index
            self.word_to_index = json.load(open(dict_path, 'r')) 
            print("keys loaded successfully from", dict_path)
        else:
            print("JSON file does not exist.")

    def create_word_keys(self, corpus: list, min_count=MIN_COUNT):
        """
        creates a vocabulary from the corpus and assigns an index to each word.

        Args:
            corpus (list): The corpus to train the model on.
            min_count (int, optional): The minimum count for a word to be included in the vocabulary. Defaults to MIN_COUNT.
        """
        vocab = set()
        for sentence in corpus:
            vocab.update(sentence)
        
        # create a dictionary of word to index
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        print("word pairs created successfully.")
    
    def save_pairs(self, dict_path):
        """
        Saves word to vocabulary index pairs to disk.
        
        Args:
            dict_path (str): The path to save the pairs to.
        """
        with open(dict_path, 'w') as f:
            json.dump(self.word_to_index, f)
        print("JSON saved successfully to: ", dict_path)
    
    def one_hot_encode(self, word:str):
        """
        Returns the one-hot encoding for a given word based on its vocab index in the word_to_index dictionary.
        Zero index is reserved for padding token.

        Args:
            word (str): The word to encode.
        
        Returns:
            torch.Tensor: A one-hot encoded tensor representing the word.
        """
        if self.word_to_index is None:
            raise Exception("key pairs is not loaded.")
        if word == PAD_TOKEN:
            return torch.nn.functional.one_hot(torch.tensor(0), num_classes=len(self.word_to_index) + 1).float()
        if word in self.word_to_index:
            vector_index = self.word_to_index[word] + 1  # +1 to accommodate padding token offset
            return torch.nn.functional.one_hot(torch.tensor(vector_index), num_classes=len(self.word_to_index) + 1).float()
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
            Exception: If the word to index dict is not loaded.
            ValueError: If the one-hot encoding does not correspond to any word in the vocabulary.
        """
        if self.word_to_index is None:
            raise Exception("Word to index pairs are not loaded.")
        
        if one_hot_vector.size()[0] != len(self.word_to_index) + 1:
            raise ValueError("Input tensor is not a valid one-hot encoding.")

        # Check if input tensor is one-hot encoded
        if torch.sum(one_hot_vector) != 1:
            raise ValueError("Input tensor is not a valid one-hot encoding.")

        # find index of 1 in one-hot vector
        index = torch.argmax(one_hot_vector).item()

        if index == 0:
            return PAD_TOKEN

        # Map index back to word, considering the offset for padding token
        word = list(self.word_to_index.keys())[index - 1]  # -1 to accommodate padding token offset
        return word

    def index_to_word(self, index: int) -> str:
        """
        Returns the word corresponding to a given index in word to index dictionaries vocabulary.

        Args:
            index (int): The index of the word to retrieve.

        Returns:
            str: The word corresponding to the index.

        Raises:
            Exception: If the word to index pairs are not loaded.
            ValueError: If the index does not correspond to any word in the vocabulary.
        """
        if self.word_to_index is None:
            raise Exception("word to index pairs are not loaded.")
        if index == 0:
            return PAD_TOKEN
        if index - 1 < len(self.word_to_index):
            return list(self.word_to_index.keys())[index - 1]
        else:
            raise ValueError("Index does not correspond to any word in the vocabulary.")

class SentenceEncodingDataset(Dataset):
    """
    A custom PyTorch Dataset for preparing sentences with encodings for training.

    Attributes:
        sentences (List[List[str]]): A list of tokenized sentences.
        encoding_manager (Any): An encoding manager instance with a method 'one_hot_encode'.
        max_sequence_length (int): The maximum length of a sequence.
        verbose (bool): Whether or not to print debug information.

    """
    def __init__(self, sentences: list, encoding_manager: EncodingManager, sequence_length: int, verbose: bool=False):
        self.sentences = sentences
        self.encoding_manager = encoding_manager
        self.sequence_length = sequence_length
        self.verbose = verbose
    
    def __len__(self):
        """Returns the number of sentences in the dataset."""
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Retrieves the encodings for the sentence at the specified index in the dataset.

        Args:
            idx (int): The index of the sentence to retrieve.

        Returns:
            torch.Tensor: A tensor of one hot encodings for the sentence.
        """
        sentence = self.sentences[idx]
        sentence = sentence[:self.sequence_length] + [PAD_TOKEN] * (self.sequence_length - len(sentence))
        if self.verbose:
            print(f'Encoding sentence: {sentence}, with encoding method: {self.encoding_method}')
        # get one hot encodings for each word in the sentence
        one_hot_encodings = [self.encoding_manager.one_hot_encode(word) for word in sentence]
        # stack the one hot encodings into a tensor
        return torch.stack(one_hot_encodings, dim=0)
    
def create_encoding_dataloader(sentences: List[List[str]], encoding_manager: Any, seq_length: int, batch_size: int, verbose: bool=False) -> DataLoader:
    """
    Creates a DataLoader for sentences with one hot encodings.

    Args:
        sentences (List[List[str]]): The tokenized sentences.
        encoding_manager (Any): An instance of the encoding manager to retrieve word encodings.
        seq_length (int): The maximum sequence length of sentences.
        batch_size (int): The number of items per batch.
        verbose (bool, optional): Whether or not to print debug information. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader object ready to be used in a training or evaluation loop.
    """
    dataset = SentenceEncodingDataset(sentences, encoding_manager, seq_length, verbose=verbose)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

