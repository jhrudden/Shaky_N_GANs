from typing import List
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import nltk 
import os
from collections import Counter

import ngram
from embeddings import WordEmbeddingManager

nltk.download('punkt')

def load_decode_and_split_shakespeare(split):
    # Load the specified split of the dataset
    dataset = tfds.load(name='tiny_shakespeare', split=split)

    # Decode byte strings to UTF-8
    dataset = dataset.map(lambda x: tf.strings.unicode_decode(x['text'], 'UTF-8'))

    # Convert the decoded Unicode code points back to strings
    dataset = dataset.map(lambda x: tf.strings.reduce_join(tf.strings.unicode_encode(x, 'UTF-8')))

    # Extract as numpy arrays and convert to Python strings
    decoded_texts = [t.decode('utf-8') for t in dataset.as_numpy_iterator()]

    # Split texts into sentences using NLTK
    sentences = [sentence for text in decoded_texts for sentence in text.split('\n') if len(sentence) > 0]

    return sentences

def tokenize_sentences(sentences: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of sentences into words.

    Args:
    sentences (List[str]): A list of sentences to tokenize.

    Returns:
    List[List[str]]: A list of tokenized sentences.
    """
    # Tokenize sentences into words using NLTK
    return [nltk.word_tokenize(sentence) for sentence in sentences]

def load_sentences(data_path: str) -> List[str]:
    """
    Loads sentences from a file.

    Args:
    data_path (str): The file path from which to load sentences.

    Returns:
    List[str]: A list of sentences loaded from the file.
    """
    assert os.path.isfile(data_path)
    with open(data_path, 'r') as f:
        sentences = f.readlines()
    return sentences

def process_data(file_path: str, min_sentence_length = 1, add_unks: bool = True) -> List[List[str]]:
    """
    Processes the data by tokenizing sentences and replacing infrequent words with "<UNK>" if applicable.

    Args:
    file_path (str): The path to the file containing the training data.
    min_sentence_length (int): The minimum length of a sentence to include in the training data. Defaults to 1.
    add_unks (bool): Whether or not to replace infrequent words with "<UNK>". Defaults to True.
    
    Returns:
    List[List[str]]: A list of tokenized sentences with infrequent words replaced by "<UNK>".
    """
    sentences = load_sentences(file_path)
    tokenized_sentences = tokenize_sentences(sentences)

    tokenized_sentences = [sentence for sentence in tokenized_sentences if len(sentence) >= min_sentence_length]

    # Count word occurrences
    word_counts = None
    if add_unks:
        word_counts = Counter(word for sentence in tokenized_sentences for word in sentence)

    processed_sentences = tokenize_sentences
    if add_unks:
        # Replace rare words with UNK
        processed_sentences = []
        for sentence in tokenized_sentences:
            processed_sentence = [word if word_counts[word] >= ngram.UNKNOWN_THRESHOLD else ngram.UNK for word in sentence]
            processed_sentences.append(processed_sentence)

    return processed_sentences

def process_non_training_data(file_path: str, word_embedding_manager: WordEmbeddingManager) -> List[List[str]]:
    """
    Processes non-training data (such as validation or test data) by tokenizing sentences and 
    replacing words not in the trained Word2Vec model's vocabulary with "<UNK>".

    Args:
    file_path (str): The path to the file containing the non-training data.
    word_embedding_manager (WordEmbeddingManager): An instance of WordEmbeddingManager containing 
    the trained Word2Vec model.

    Returns:
    List[List[str]]: A list of tokenized sentences with words not in the model's vocabulary replaced by "<UNK>".
    """
    sentences = load_sentences(file_path)
    tokenized_sentences = tokenize_sentences(sentences)

    # Retrieve the vocabulary from the Word2Vec model
    trained_vocab = set(word_embedding_manager._model.wv.key_to_index.keys())

    # Process sentences
    processed_sentences = []
    for sentence in tokenized_sentences:
        processed_sentence = [word if word in trained_vocab else ngram.UNK for word in sentence]
        processed_sentences.append(processed_sentence)

    return processed_sentences
