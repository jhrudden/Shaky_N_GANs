import ngram
from gensim.models import Word2Vec


EMBEDDING_SIZE = 50
WINDOW_SIZE = 5
MIN_COUNT = ngram.UNKNOWN_THRESHOLD

class WordEmbeddingManager:
    _model = None
    def __init__(self, model_path:str = None):
        self._model = None
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path):
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
    


