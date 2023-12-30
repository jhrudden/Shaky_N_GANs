### General Idea:

I was inspired by Andre Karpothy’s series for building A GPT model to generate Shakespeare lines. Since we were already covering the makeup of transformers, I thought it would be interesting to investigate a different model. Recently, I have been hearing a ton about GANs and the general idea of this approach (which I will talk about later) seemed super interesting. So, for this project I decided to build a GAN for sequence generation. As a baseline to compare outputs generated from my GAN to outputs from a generative model that we have seen, I decided to also train a NGRAM model which uses linear interpolation to improve the probability of output sequences, thus, NGRAM should output some quality sequences that can benchmark my GANs outputs.

### It’s all about the Data:

As previously mentioned, the goal of this project is to build a fairly complex model that can generate Shakespeare-esque lines. Moreover, I have decided to use the Tiny-Shakespeare dataset for my training data. This corpus was created by Andre Karpathy as training data for a character RNN in his blog The Unreasonable Effectiveness of Recurrent Neural Networks. The training portion of the dataset (which I use for both training and validation) contains roughly 29k lines of Shakespeare with roughly 229k tokens. The validation portion contains ~1700 lines and 13k tokens. Each line of this dataset is taken directly from one of Shakespeare's works.

### Perplexity:

One of the most popular measures used for evaluating models and a metric we have seen a bunch in class is perplexity. Perplexity measures the normalized inverse probability of each word in a sequence. The formula for perplexity is as follows:

$$
perplexity(W) = \sqrt[N]{\prod_i^{N} \frac{1}{P(w_i | w_{1}...w_{i-1})}}
$$

Where $P(w_i | w_{1}...w_{i-1})$ represents the probability of next word $w_i$ given the previous context output by the model, $W$ represents a sequence of words, and $N$ represents the sequence length.

Moreover, higher perplexity scores means that a sequence was less probable. Our goal for training these models is that they output sequences with high probabilities, as this would hopefully mean they are similar to the training data. It is important to note that just because a model outputs a low perplexity score (which we want), doesn't mean that these sequences are all that spectacular.

Regardless of this limitation, I will use perplexity (specifically perplexity calculated by my NGRAM) to evaluate both models.

### NGRAMs:

As we have seen in class, the NGRAM language model uses a little trick to approximate the probability of the next word. An example of this function for bigrams might be:

$$
	P(w_n | w_{1:n-1}) \approx P(w_n | w_{n-1})
$$

For trigrams it would look like:

$$
	P(w_n | w_{1:n-1}) \approx P(w_n | w_{n-1}, w_{n-2})
$$

And so on.
Moreover, this assumption is that the next word in a sequence only depends on the previous $n-1$ words. Building on this, for a sequence of words, NGRAMs approximates the probability of that sequence occurring as follows: (bigram example)

$$
P(W) = \prod_i^N P(w_i | w_{i-1})
$$

Where:

$$
P(w_i | w_{i-1}) = \frac{c(w_{n-1}, w_n)}{c(w_{n-1})}
$$

I don't want to get too into the weeds here, but for Lab1 we to avoid zero probabilities for sequences we haven't used before, we can use what is called Laplace smoothing. This follows the formula (for trigrams):

$$
P(w_n | w_{n-1:n-2}) = \frac{c(w_{n-2}, w_{n-1} , w_n) + 1}{c(w_{n-2}, w_{n-1}) + |V|}
$$

Where $|V|$ is the size of the vocabulary of training corpus. By looking at this equation you can see that unseen NGRAMs are given a fairly high probability.

For this reason (and cause Felix said I couldn't use Laplace smoothing) I used a different method to smooth my NGRAM probabilities _Linear Interpolation_.

Linear interpolation works by calculating a probability of a given NGRAM by calculating a weighted sum of probabilities of its sub grams. This follows the formula (for trigram):

$$
\hat{P}(w_n | w_{n-2|n-1}) = \lambda_1P(w_n) + \lambda_2P(w_n | w_{n-1}) + \lambda_2P(w_n | w_{n-2|n-1})
$$

Moreover, for an NGRAM model with Linearly interpolated, we actually need a unigram, bigram and trigram model too. Each of these models don't employ a smoothing technique, so if their probability is zero, then they just don't contribute to that NGRAMs probability. Assuming the model is is always fed sequences with known unigrams or $<UNK>$ tokens, then probabilities of a sequence will never be zero. However, All $\lambda$ values must sum to 1.

$$
1 = \sum_i^n \lambda_i
$$

There are various ways to find the optimal $\lambda$ values. For the sake of simplicity I decided to use grid search and selected values using Maximum Likelihood Estimation on a held-out corpus. A held-corpus is a corpus taken from the training dataset. It is taken without replacement, so actual NGRAM model never sees values in the held-out corpus during training. Once again, I selected $\lambda$ that resulted in the high average likelihood of sequences in the held-out corpus.

### Training NGRAM Time:

To help gauge the improvements of a NGRAM using Linear Interpolation over Laplace smoothing, I trained both models. I also collected metrics on how NGRAM size (e.g. unigram vs bigram vs ...) affected perplexity scores. Similarly, I also collected data on how train set size affected model performance (I trained models on 0.2, 0.4 ,...,1 of the train dataset). Here are some visuals of the results:

#### Affects of NGRAM Size on Perplexity:

![[https://github.com/jhrudden/Shaky_N_GANs/figs/laplace_perplexity_sizes.png|400]]
![[https://github.com/jhrudden/Shaky_N_GANs/figs/li_perplexity_sizes.png|400]]

Moreover, we see that Linear Interpolation seems perform better in all cases. Also it seems to generally be the case that more complex NGRAMs perform better on the perplexity metric. Not the same for Laplace.

### Results From Varying Train Proportion:

![[https://github.com/jhrudden/Shaky_N_GANs/figs/laplace_perplexity_train_size.png|400]]
![[https://github.com/jhrudden/Shaky_N_GANs/figs/li_perplexity_train_size.png|400]]
![[https://github.com/jhrudden/Shaky_N_GANs/figs/li_vs_laplace_perplexity_train_size.png|400]]

The above results seem to show what was originally mentioned about high weighting of unknown probabilities for Laplace smoothing. A larger train set means larger vocabulary, which Laplace doesn't seem to deal well with. On the other hand, the LI model seems to stay in the same ballpark when it comes to perplexity changes based on train set size, which is an interesting and good sign.

### How about NGRAM Shakespeare?

Enough with the quantitative stuff, time to see how well these models generate text. For generation, I chose to use 4GRAMs because it was quick to train and perplexity scores weren't super different from the 5GRAMs I had previously trained. I generated 3 sample sentences from both models.

### Generated Sentences Laplace:

```
cracking the stones of the <UNK>, 't was nothing;

Not mad, but bound more than a <UNK> is Hastings, that he hath two,

several tunes faster than you'll tell money; he
```

### Generated Sentences LI

```
Till that my nails were anchor'd in thine eyes;

Yourself, your queen, your son was gone before I came.

wicked varlet, now, how a jest shall come about!
```

## Generative Adversarial Networks

The core concept of a Generative Adversarial Network (GAN) involves learning a new distribution $q(x)$ that closely approximates a given distribution $p(x)$, based on independent and identically distributed (IID) samples. This process requires two key components: the _Generator_ and the _Discriminator_.

1. **Generator**: The Generator's task is to take samples $z$ from a known distribution (like $z \sim U(0,1)$ or $z \sim N(0,1)$) and transform them into samples that mimic those from $p(x)$. The Generator generates fake data aiming to pass as real.
2. **Discriminator**: Conversely, the Discriminator's role is to distinguish whether input sequences are from the original distribution $p(x)$ or the generated distribution $q(x)$, where $q(x) = G(z)$ and $G(z)$ denotes the output from the Generator. The Discriminator essentially acts as a judge, determining the authenticity of the samples.

The interaction between these two components creates a dynamic competition: the Generator works to produce increasingly convincing fakes, while the Discriminator becomes better at detecting them. This competitions drives the improvements to each model.

### GAN Objectives

In a typical GAN, the Discriminator is a binary classifier aiming to label real data as 1 and generated data as 0. The loss functions used are:

##### Discriminator Objective

$$
\text{D\_Loss} = -\frac{1}{n}\sum_i^n y_ilog(D(x_i)) + (1-y_i)log(1-D(G(z_i)))
$$

##### Generator Objective

$$
\text{G\_Loss} = \frac{1}{n}\sum_i^n y_ilog(D(G(z_i)))
$$

### Data Representation in GANs for Text Generation

In the context of my Generative Adversarial Network (GAN) model, "real" data, denoted as $x$, comprises tokens from sentences in the Shakespeare training set. The Discriminator will process entire sentences as sequences of these samples, evaluating their authenticity. Each token $x$ represents a sample from a distribution, essentially modeling the probability of the next word given some preceding context.

To capture the sequential and contextual nature of text, both the Generator and Discriminator incorporate Long Short-Term Memory (LSTM) layers. LSTMs are adept at handling sequences, allowing the model to consider the context and dependencies between words. The Generator, in particular, will generate a sequence of predictions each time it runs, rather than individual, isolated words.

For data representation, individual tokens from the training set are expressed as one-hot encodings. In this scheme, each token's position in the vocabulary determines its unique encoding. This approach allows for a clear and distinct representation of each word in the training data.

In the language models discussed in class, text generation involves sampling each word from a joint distribution, informed by the context established by preceding words. However, this approach of direct sampling poses a significant challenge in a GAN framework. Direct sampling from a softmax distribution, which we often use due to the multinomial nature of word choice, presents a significant challenge in GANs due to its non-differentiable nature
This limitation becomes particularly problematic as it interrupts the flow of gradients during backpropagation, this process is required for training the Generator based on feedback from the Discriminator.

There are a few tricks you can do here to work around this roadblock. Some include using Reinforcement Learning, but that kind felt like a whole can of worms to me, so I chose to do a different clever trick known as the Gumbel Softmax (also felt a little easier).

## Gumbel Softmax

Consider a multinomial distribution pp representing some value x, derived from continuous logits h output from our Generator:

$$p=softmax(h)$$

Traditionally, sampling from this distribution would be non-differentiable which creates an issue for our GAN.

This process can equivalently represented as selecting a value y using the following method and introducing a noise vector g sampled from a Gumbel distribution (with the same dimensions as h):
$$y=one\_hot(argmax_i(h_i+g_i))$$
This approach, while insightful, still presents a non-differentiable operation due to the argmax function. However, some super smart researchers developed a clever workaround to approximate this process in a differentiable way.

By adding a temperature scalar α, y can be approximated by:
$$y=softmax(\frac{1}{α}(h+g))$$
Here, α plays a crucial role. As α approaches 0, y increasingly resembles a categorical sample, effectively mimicking the argmax operation. Conversely, as α grows larger, approaching infinity, the distribution of y becomes more uniform.

This process allows us to 'soften' the traditionally hard decision-making process of sampling discrete tokens, allowing us to back-propagate! By applying this technique to the output logits of our Generator, the sequences generated can approximate samples drawn from the softmax(h) distribution. This enables effective training of the Generator within the GAN!

## Training Our GAN:

For the training of my Seq GAN, I employed batch training. Given the considerable size of my models, training for multiple epochs was not feasible on my local machine. Depending on the hyperparameters, my Generator averaged around 2.5 million parameters, with the Discriminator having a slightly higher count.

The main hyperparameters I focused on tuning were the learning rate, the temperature parameter for the Gumbel Softmax, the sequence lengths of real/generated data, and the batch size. To accommodate different sequence lengths, I either truncated or padded the real data to align with the sequence length hyperparameter.

My analysis of the optimal hyperparameters was largely based on a qualitative evaluation of the sentences generated by the trained GAN. However, I also monitored BLEU scores and perplexity to quantify the performance.

An important aspect of the training process was managing the balance between the Discriminator and Generator. I closely tracked the losses for both, aiming to prevent the Discriminator from learning too quickly and overly penalizing the Generator. This was crucial to maintain a competitive training environment. Additionally, I monitored the Discriminator's accuracy, recall, precision, and F1 score. For the Generator, I focused on its accuracy in deceiving the Discriminator, ensuring that it remained a challenging adversary.

### Note on Perplexity:

Due to the nature of the Generator's output in my GAN, which utilizes a Gumbel Softmax (an approximation of sampling from a multinomial distribution), calculating the perplexity of the generated sequences directly is not straightforward. The model does not produce probability distributions in a form that allows for easy computation of joint probabilities. To address this challenge, I devised a method that involves leveraging a trained NGRAM model on the same training set.

By inputting the sentences generated by the GAN into the NGRAM model's perplexity function, I can approximate the perplexity of these sequences. It's important to note, however, that this method is not an unbiased estimator of perplexity. The reason for this bias is the difference in context consideration between the two models: while the GAN's LSTM-based architecture allows it to consider the entire contextual sequence, the NGRAM model only evaluates joint probabilities based on smaller sub-windows of the input. Despite this limitation, I believe this approach provides a reasonable, albeit biased, estimate of the GAN's perplexity for our current purposes.

## Training Results

Due to the constrained space of this report, it's not plausible to present the outcomes of all hyperparameter experiments in detail. However, it's worth noting that identifying the optimal combination of hyperparameters was rough, especially given the complexity involved in generating text, particularly text mimicking Shakespeare's style.

The majority (maybe even all) model iterations struggled to generate meaningful content. A notable observation was the model's fixation on sequence length. Often, the models would generate just two words and then fill the remainder of the input with padding. This behavior can be partly explained by examining the sequence length distribution in the training data:
![[https://github.com/jhrudden/Shaky_N_GANs/figs/seq_length_distribution.png]]
As evident from the plot, a significant portion of the training data sequences are of length two, which likely influenced the GAN's training behavior.

## Best Results

![[https://github.com/jhrudden/Shaky_N_GANs/figs/gan_loss_plot.png]]
![[https://github.com/jhrudden/Shaky_N_GANs/figs/gan_perplexity_plot1.png]]
The above plot illustrates the loss of both the Generator and Discriminator over the course of training in my most successful hyperparameter tuning session. A constant competition is observable between the two components, which is a positive sign. However, towards the end, the Generator's loss starts to diverge, suggesting that the Discriminator may have become overly proficient.

Additionally, the log-scaled perplexity scores during the first 300 batches of training showed an initial score in the 100,000 range, eventually decreasing to around 100. This indicates some level of learning and adaptation by the model.

Despite these promising signs, the qualitivative results were less than satisfactory. Here are three sentences generated by the GAN:

```
1. "breeding they escaped"
2. "Francis loathed faster"
3. "stripes avoid leads forces petty spurn target faction throws sire virtuous poison Volsces swan County"
```

These outputs leave much to be desired in terms of coherence and literary quality. Even a simple NGRAM model managed to generate more coherent sentences. This outcome was somewhat anticipated, as the limitations of my computing resources restricted training to no more than one epoch. Nevertheless, it is disappointing to observe the incoherence in the results. Perhaps, this reflects the intricate complexity and beauty of Shakespeare's work that remains a challenge for AI models to capture.

## Future Directions: Improvements for Next Time

Reflecting on the challenges and learnings from this project, there are a few strategic adjustments I would consider for future iterations:

**Exploring Alternative Sampling Methods**: One of the key issues encountered was related to the use of Gumbel Softmax. While it provided a workaround for the non-differentiable nature of traditional sampling methods, I have seen many comments favoring alternative methods. In future projects, I might explore other techniques, such as Reinforcement Learning, to handle the sampling process.

**Extended Training Periods**: The significant size of the models (2.5 million parameters for the Generator and even more for the Discriminator) posed a substantial computational challenge. Training for even one epoch on the training set required 30-40 minutes, often leading to crashes on my local machine. Recognizing the need for more extensive training to improve model performance, I would consider training such models on rented hardware.

Thank you for reading!
