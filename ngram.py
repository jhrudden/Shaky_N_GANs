from collections import Counter
import numpy as np  
import nltk
from typing_extensions import Literal

def tokenize(text: str, n: int) -> list:
    """
    Tokenize a text for ngram model.
    This requires padding each sentence with SENTENCE_BEGIN and SENTENCE_END tokens.
    Args:
      text (str): a string representing a sequence of sentences
    """
    lines = nltk.sent_tokenize(text)
    sentences = []
    for line in lines:
        sentence = []
        if n > 1:
            sentence = [SENTENCE_BEGIN] * (n - 1) + nltk.word_tokenize(line) + [SENTENCE_END] * (n - 1)
        else:
            sentence = [SENTENCE_BEGIN] + nltk.word_tokenize(line) + [SENTENCE_END]  
        
        sentences.append(sentence)
    
    return sentences


def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
      tokens (list): a list of tokens as strings
      n (int): the length of n-grams to create

    Returns:
      list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))
    return ngrams

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

UNKNOWN_THRESHOLD = 2

class NGRAM_Model:
    n = None
    n_gram_frequencies = Counter()
    n_minus_one_gram_frequencies = Counter()
    number_training_tokens = 0
    number_of_non_starting_tokens = 0
    vocab_size = 0
    vocab = set()

    def __init__(self, n_gram):
        """Initializes an untrained NgramModel instance.
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        self.n = n_gram
   
    def perplexity(self, sentence_tokens: list) -> float:
        """Calculates the perplexity of a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a list of tokens to calculate the perplexity of

        Returns:
          float: the perplexity value of the given tokens for this model
        """
        # ensure sentence doesn't contain any tokens that are not in the vocab
        updated_sentence_tokens = [
            UNK if token not in self.vocab else token for token in sentence_tokens
        ]

        # begin scoring
        ngrams = create_ngrams(updated_sentence_tokens, self.n)
        probability = 1
        for ngram in ngrams:
            # Apply laplace smoothing to avoid zero probabilities
            probability *= 1 / self.score_ngram_laplace(ngram)
        return probability ** (-1 / len(ngrams))

    def train(self, train_tokens: list, held_out_tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          train_tokens (list): tokenized data to be trained on as a single list
          held_out_tokens (list): tokenized data to be used for finding linear interpolation weights
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        # convert all tokens with counts less than UNKNOWN_THRESHOLD to UNK
        token_counts = Counter(tokens)
        keys_to_remove = [
            key for key, count in token_counts.items() if count < UNKNOWN_THRESHOLD
        ]
        updated_tokens = [UNK if token in keys_to_remove else token for token in tokens]

        # get frequencies of ngrams and n-1 grams for scoring
        ngrams = create_ngrams(updated_tokens, self.n)
        n_minus_one_grams = create_ngrams(updated_tokens, self.n - 1)
        self.n_gram_frequencies = Counter(ngrams)
        self.n_minus_one_gram_frequencies = Counter(n_minus_one_grams)

        # get frequencies of tokens for scoring of unigrams
        self.number_training_tokens = len(updated_tokens)
        self.number_of_non_starting_tokens = len(
            list(filter(lambda x: x != SENTENCE_BEGIN, updated_tokens))
        )

        # get vocab size for laplace smoothing
        self.vocab = set(updated_tokens)
        self.vocab_size = len(self.vocab)

        if verbose:
            print("top 10 ngram_frequencies: ", self.ngram_frequencies.most_common(10))
            print("top 10 token_freqencies: ", self.token_frequencies.most_common(10))
            print("total ngrams: ", len(self.ngram_frequencies))
 
    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model

        Returns:
          float: the probability value of the given tokens for this model
        """

        # ensure sentence doesn't contain any tokens that are not in the vocab
        updated_sentence_tokens = [
            UNK if token not in self.vocab else token for token in sentence_tokens
        ]

        # begin scoring
        ngrams = create_ngrams(updated_sentence_tokens, self.n)
        probability = 1
        for ngram in ngrams:
            # Apply laplace smoothing to avoid zero probabilities
            probability *= self.score_ngram_laplace(ngram)
        return probability

    def score_ngram_laplace(self, ngram: tuple) -> float:
        """
        Calculates the probability score for a given ngram of size n. Use laplace smoothing to avoid zero probabilities
        Args:
          ngram (tuple): a tuple of strings representing an ngram
        """
        def get_denom(ngram):
            if self.n > 1:
                return self.n_minus_one_gram_frequencies[ngram[:-1]]
            else:
                return (
                    self.number_training_tokens
                )

        assert len(ngram) == self.n

        return (self.n_gram_frequencies[ngram] + 1) / (
            get_denom(ngram) + self.vocab_size
        )

    def score_ngram_generation(self, ngram: tuple) -> float:
        """
        Calculates the probability score for a given ngram of size n. Assume generated ngrams
        must have been seen in training data
        Args:
          ngram (tuple): a tuple of strings representing an ngram
        """
        assert len(ngram) == self.n
        assert ngram in self.n_gram_frequencies

        # if ngram is a unigram, we will not being generating start tokens, so we only look at training tokens that are not start tokens
        denominator = (
            self.n_minus_one_gram_frequencies[ngram[:-1]]
            if self.n > 1
            else self.number_of_non_starting_tokens
        )

        return self.n_gram_frequencies[ngram] / (denominator)

    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          list: the generated sentence as a list of tokens
        """
        sentence = []
        # Add necesary number of sentence begin tokens
        for _ in range(self.n - 1):
            sentence.append(SENTENCE_BEGIN)
        if self.n == 1:
            sentence.append(SENTENCE_BEGIN)

        while sentence[-1] != SENTENCE_END:
            # assume we have a trained unigram model
            all_possible_ngrams = []
            all_possible_ngrams_probabilities = []

            if self.n == 1:
                # For unigrams, we can just use the vocab - SENTENCE_BEGIN token as the possible ngrams at any point
                # This is because we don't want to see a sentence begin token in the middle of a sentence

                all_possible_ngrams = [
                    (word,)
                    for word in self.vocab.symmetric_difference({SENTENCE_BEGIN})
                ]

            else:
                # need list of all ngrams that start with the last n-1 words of the current sentence
                current_n_minus_one_gram = tuple(sentence[-(self.n - 1) :])
                all_possible_ngrams = [
                    ngram
                    for ngram in self.n_gram_frequencies.keys()
                    if ngram[:-1] == current_n_minus_one_gram
                ]


            all_possible_ngrams_probabilities = [
                self.score_ngram_generation(ngram) for ngram in all_possible_ngrams
            ]


            # randomly sample next ngram based on probabilities
            next_ngram_index = np.random.choice(
                len(all_possible_ngrams), p=all_possible_ngrams_probabilities
            )

            next_ngram = all_possible_ngrams[next_ngram_index]
            # add the last word of the selected ngram to the sentence
            sentence.append(next_ngram[-1])

        # make sure sentences end with n-1 sentence end tokens
        if self.n > 1:
            sentence += [SENTENCE_END] * (self.n - 2)
        return sentence

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        return [self.generate_sentence() for i in range(n)]

class LinearInterpolation:
    def __init__(self, ngram: int, held_out_tokens: list, method: Literal['grid_search', 'expectation_maximization'], sub_divisions: int = 50):
        self.ngram = ngram
        ngram_to_freq = {}
        num_tokens = len(held_out_tokens)
        self.ngram_to_prob = {}
        self.ngrams = {}
    
        for n in range(1, ngram + 1):
            curr_ngrams = create_ngrams(held_out_tokens, n)
            if n == self.ngram:
                self.ngrams = curr_ngrams
            ngram_to_freq[n] = Counter(curr_ngrams)

            if n > 1:
                self.ngram_to_prob[n] = {k: v / ngram_to_freq[n-1][k[-n:][:-1]] for k, v in ngram_to_freq[n].items()}
            else:
                self.ngram_to_prob[n] = {k: v / num_tokens for k, v in ngram_to_freq[n].items()}

        # turn ngram_to_prob into matrix where each row is a sample from self.ngrams and each column is a probability of col+1 gram occuring given the previous col grams
        self.X = np.zeros((len(self.ngrams), self.ngram))
        for i, ngram in enumerate(self.ngrams):
            for j in range(self.ngram):
                self.X[i, j] = self.ngram_to_prob[j+1][ngram[-(j+1):]]
        
        self.method = method
        self.sub_divisions = sub_divisions
        self.weights = None
        
    def train(self):
        if self.weights is not None:
            pass
        if self.method == 'grid_search':
            self.grid_search(self.sub_divisions)
        elif self.method == 'expectation_maximization':
            self.expectation_maximization()
        else:
            raise ValueError(f'Unknown method {self.method}')

        
    def grid_search(self, sub_divisions: int):
        weights = np.ones(self.ngram) / self.ngram
        best_mle = -float('inf')
        # try self.ngram - 1 weights for each weight
        lambdas_combinations = self._generate_possible_lambda_combinations(self.ngram, sub_divisions)
        # choose self.ngram weights that sum to 1 needs to work for ngrams > 3
        for lambdas in lambdas_combinations:
            mle = self._log_likelihood(lambdas)
            if mle > best_mle:
                best_mle = mle
                best_weights = lambdas
        
        self.weights = best_weights
        return best_weights
    
    def draw_samples(self):
        # draw samples from self.ngrams
        # create self.ngram plots side by side of the distribution of each ngram
        fig, axs = plt.subplots(1, self.ngram, figsize=(10, 10))
        for i in range(self.ngram):
            axs[i].hist(self.X[:, i], bins=50)
            axs[i].set_title(f'Ngram {i+1}')
        plt.show()
    

    def expectation_maximization(self):
        # initialize weights to be uniform
        weights = np.ones(self.ngram) / self.ngram
        # iterate until convergence
        while True:
            expectations = self._e_step(weights)
            new_weights = self._m_step(expectations)
            if np.allclose(weights, new_weights):
                break
            weights = new_weights
        self.weights = weights

    def _e_step(self, weights):
        # TODO: Shouldn't this use MLE? Right now we are determining how weighted ngram probs individually contribute to overal sum of weighted ngram probs
        # But do we also need to add some non-linearities to the weights? e.g. MLE?
        gram_weight_prods = self.X * weights 
        gram_impacts_per_sample = gram_weight_prods / np.sum(gram_weight_prods, axis=1).reshape(-1, 1)
        return gram_impacts_per_sample.sum(axis=0)

    def _m_step(self, expectations):
        return expectations / np.sum(expectations)

    def _log_likelihood(self, lambdas):
        # want np.log to ignore 0s
        product = self.X * lambdas
        non_zero_mask = product != 0
        log_product = np.zeros(product.shape)
        log_product[non_zero_mask] = np.log(product[non_zero_mask])

        return -np.sum(log_product)
    
    def _generate_possible_lambda_combinations(self, dimensions: int, sub_divisions: int):
        # generating points along a simplex whos vertices are (0, 0, ..., 1), (0, 0, ..., 1), ..., (1, 0, ..., 0)
        def recursive_generation(current_points, remaining, depth):
            if depth == dimensions - 1:
                yield current_points + [remaining]
            else:
                for i in range(sub_divisions - sum(current_points)):
                    yield from recursive_generation(current_points + [i], remaining - i, depth + 1)
        
        scale = 1 / sub_divisions
        return list(map(lambda x: list(map(lambda y: y * scale, x)), recursive_generation([], sub_divisions, 0)))
