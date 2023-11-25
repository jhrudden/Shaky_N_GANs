import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import nltk 
import os
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

def tokenize_sentences(sentences):
    # Tokenize sentences into words using NLTK
    return [nltk.word_tokenize(sentence) for sentence in sentences]

def load_sentences(data_path: str):
    assert os.path.isfile(data_path)
    with open(data_path, 'r') as f:
        sentences = f.readlines()
    return sentences