# Shaky_N_GANs

### TODO:

-   [] When training make sure that the subgram frequencies are based on a tokenized version of train data based on that subgram.
    So if I get some data X for a trigram, then I will need
-   X tokenized for a unigram
-   X tokenized for a bigram
-   X tokenized for a trigram

Right now I am just using the tokenized version of the trigram for all of them. This is not correct as the distrubution of the unigram, bigram, will include all start token subgrams.
