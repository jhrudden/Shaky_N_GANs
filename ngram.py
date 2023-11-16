from collections import Counter
import numpy as np  
import nltk
from typing_extensions import Literal
import matplotlib.pyplot as plt

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"

UNKNOWN_THRESHOLD = 2

def tokenize(text: str, n: int) -> list:
    """
    Tokenize a text for ngram model.
    This requires padding each sentence with SENTENCE_BEGIN and SENTENCE_END tokens.
    Args:
      text (str): a string representing a sequence of sentences
    """
    # lines = nltk.sent_tokenize(text)
    lines = text.split('\n')
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

class NGRAM_Model:
    n = None
    number_training_tokens = 0
    number_of_non_starting_tokens = 0
    vocab_size = 0
    vocab = set()

    def __init__(self, n_gram, scoring_method: Literal['laplace', 'LI_expectation_maximization', 'LI_grid_search'] = 'laplace'):
        """Initializes an untrained NgramModel instance.
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        self.n = n_gram
        self.scoring_method = scoring_method
        if "LI" in scoring_method:
            method = "expectation_maximization" if "expectation_maximization" in scoring_method else "grid_search"
            self.linear_interpolation = LinearInterpolation(n_gram, method=method)
        elif scoring_method == 'laplace':
            self.linear_interpolation = None
        else:
            raise ValueError(f'Unknown scoring method {scoring_method}')

    # TODO: train should take in a list of tokenized sentences then split into train and held out depending on the model smoothing method
    def train(self, train_sentences: list, hold_out_size: float = 0.1, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          train_sentences (list): a list of tokenized sentences to train on
          hold_out_size (float): the percentage of training data to hold out for linear interpolation training
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        # split into train and held out if necessary
        if self.scoring_method != 'laplace':
            num_held_out = int(len(train_sentences) * hold_out_size)
            held_out_sentences = train_sentences[:num_held_out]
            train_sentences = train_sentences[num_held_out:]
            # TODO: maybe need to add UNK to held out sentences?
            held_out_tokens = np.concatenate(held_out_sentences)
            if verbose:
                print(f"Training with {len(train_sentences)} sentences and {len(held_out_sentences)} held out sentences")
                print("Beginning linear interpolation training")
            self.linear_interpolation.train(held_out_tokens)
            if verbose:
                print("Finished linear interpolation training. Weights: ", self.linear_interpolation._weights)
        
        train_tokens = np.concatenate(train_sentences)
        # convert all tokens with counts less than UNKNOWN_THRESHOLD to UNK
        token_counts = Counter(train_tokens)
        keys_to_remove = [
            key for key, count in token_counts.items() if count < UNKNOWN_THRESHOLD
        ]
        updated_tokens = [UNK if token in keys_to_remove else token for token in train_tokens]

        # depending on scoring method, we may need more than just the ngrams and n-1 grams
        self.gram_to_freq = {}
        if self.scoring_method == 'laplace':
            ngrams = create_ngrams(updated_tokens, self.n)
            self.gram_to_freq[self.n] = Counter(ngrams)
            if self.n > 1:
                n_minus_one_grams = create_ngrams(updated_tokens, self.n - 1)
                self.gram_to_freq[self.n - 1] = Counter(n_minus_one_grams)
        
        elif self.scoring_method == 'LI_expectation_maximization' or self.scoring_method == 'LI_grid_search':
            for n in range(1, self.n + 1):
                curr_ngrams = create_ngrams(updated_tokens, n)
                self.gram_to_freq[n] = Counter(curr_ngrams)
        


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
        Scoring method is determined by the scoring_method attribute of this model
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

        if self.scoring_method == 'laplace':
            return np.prod([self.score_ngram_laplace(ngram) for ngram in ngrams])
        else:
            return np.prod([self.score_ngram_linear_interpolation(ngram) for ngram in ngrams])

    def score_ngram_laplace(self, ngram: tuple) -> float:
        """
        Calculates the probability score for a given ngram of size n. Use laplace smoothing to avoid zero probabilities
        Args:
          ngram (tuple): a tuple of strings representing an ngram
        """
        def get_denom(ngram):
            if self.n > 1:
                return self.gram_to_freq[self.n - 1][ngram[:-1]]
            else:
                return (
                    self.number_training_tokens
                )

        assert len(ngram) == self.n

        return (self.gram_to_freq[self.n][ngram] + 1) / (
            get_denom(ngram) + self.vocab_size
        )
    
    def score_ngram_linear_interpolation(self, ngram: tuple) -> float:
        assert self.linear_interpolation is not None
        assert self.linear_interpolation._weights is not None
        assert len(ngram) == self.n

        weights = self.linear_interpolation._weights
        score = 0
        for i in range(self.n, 0, -1):
            current_gram_freq = self.gram_to_freq[i][ngram[-i:]]
            current_n_minus_one_gram_freq = self.gram_to_freq[i - 1][ngram[-i:][:-1]] if i > 1 else self.number_training_tokens
            # backoff to lower order ngrams if current ngram has zero probability
            if current_n_minus_one_gram_freq == 0 or current_gram_freq == 0:
                continue
            p_current_gram = current_gram_freq / current_n_minus_one_gram_freq
            score += weights[i-1] * p_current_gram

        # this should never happen since we are using UNK tokens, so unigram freqs should never have zero probability
        if score == 0:
            print(f"Score is 0 for ngram {ngram} with subgrams {subgrams} and sub_freqs {sub_freqs}")

        return score

    # TODO make this work for linear interpolation too
    def score_ngram_generation(self, ngram: tuple) -> float:
        """
        Calculates the probability score for a given ngram of size n. Assume generated ngrams
        must have been seen in training data
        Args:
          ngram (tuple): a tuple of strings representing an ngram
        """
        assert len(ngram) == self.n
        assert ngram in self.gram_to_freq[self.n]

        # if ngram is a unigram, we will not being generating start tokens, so we only look at training tokens that are not start tokens
        denominator = (
            self.gram_to_freq[self.n - 1][ngram[:-1]]
            if self.n > 1
            else self.number_of_non_starting_tokens
        )

        return self.gram_to_freq[self.n][ngram] / (denominator)

    # GET THIS WORKING WITH LINEAR INTERPOLATION
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
                    for ngram in self.gram_to_freq[self.n].keys()
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
        probability = self.score(updated_sentence_tokens)
        N = len(list(filter(lambda x: x != SENTENCE_BEGIN, updated_sentence_tokens)))
        return probability ** (-1 / N)

class LinearInterpolation:
    def __init__(self, ngram: int, method: Literal['grid_search', 'expectation_maximization'], sub_divisions: int = 50):
        self.ngram = ngram
        self.ngram_to_prob = {}
        self.ngrams = {}
    
        self.method = method
        self.sub_divisions = sub_divisions
        self._weights = None
        
    def train(self, held_out_tokens: list):
        if self._weights is not None:
            return

        ngram_to_freq = {}
        num_tokens = len(held_out_tokens)
        for n in range(1, self.ngram + 1):
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
        
        self._weights = best_weights
        return best_weights

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
        self._weights = weights

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
