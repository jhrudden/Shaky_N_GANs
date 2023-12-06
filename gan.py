from typing import Literal, Any, Callable, List
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from embeddings import create_embedding_dataloader
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import ngram

def load_gen_model(model_path: str, gen_input_dim: int, gen_hidden_dim: int, gen_output_dim: int, seq_length: int):
    """
    Loads a model from a file.

    Args:
        model_path (str): The path to the model file.
        gen_input_dim (int): The input dimension (number of features) for the generator.
        gen_hidden_dim (int): The hidden dimension for the LSTM blocks.
        gen_output_dim (int): The output dimension (number of features) for the generator (this will be the size of our vocab for now).
        seq_length (int): The sequence length of the generated sequences.

    Returns:
        Any: The loaded Generator model.
    """
    # Instantiate the model
    gen = Generator(input_size=gen_input_dim, hidden_size=gen_hidden_dim, output_size=gen_output_dim, seq_length=seq_length)

    # Load the state dictionary
    state_dict = torch.load(model_path)
    missing_keys, unexpected_keys = gen.load_state_dict(state_dict, strict=False)

    # Check for missing or unexpected keys
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return gen


def estimate_perplexity(sentences: List[List[str]], ngram_model):
    """
    Calculates the perplexity of a set of sentences based on an ngram model.

    Args:
        sentences (List[List[str]]): A list of tokenized sentences.
        ngram_model (Any): The ngram model to use for calculating perplexity.

    Returns:
        float: The perplexity of the sentences.
    """
    padded_sentences = [ngram.prepare_tokenized_sentences([sentence], ngram_model.n) for sentence in sentences]
    mean_perplexity = np.mean([ngram_model.perplexity(sentence) for sentence in padded_sentences])
    return mean_perplexity
    

def gumbel_softmax(logits, temperature: int, hard: bool=False):
    """
    Implements the gumbel softmax function described in https://arxiv.org/pdf/1611.04051.pdf

    Args:
        logits (torch.Tensor): The logits to apply the gumbel softmax function to.
        temperature (int): The temperature parameter used to control the smoothness of the softmax function.
        hard (bool, optional): If True, the output will be one-hot encoded. Default is False.
    """
    U = torch.rand(logits.shape)
    G = -torch.log(-torch.log(U + 1e-20) + 1e-20)


    y = F.softmax((logits + G) / temperature, dim=1)
    if not hard:
        return y.view(-1, logits.shape[1])
    
    largest_indices = torch.argmax(y, dim=1)
    one_hot = torch.zeros_like(y)
    one_hot[torch.arange(y.shape[0]), largest_indices] = 1
    # y_hard has the same gradients as y, but y_hard is one-hot
    y_hard = (one_hot - y).detach() + y
    return y_hard.view(-1, logits.shape[1])

class Generator(nn.Module):
    """
    Generator component of the GAN. Generates a sequence of probabilities for each word in the vocabulary based on the input noise.

    Attributes:
        input_size (int): The input dimension (number of features) for the generator.
        output_size (int): The output dimension (number of features) for the generator (this will be the size of our vocab for now).
        hidden_size (int): The hidden size for the LSTM blocks.
        seq_size (int): The sequence length of the generated sequences.
        num_lstm_blocks (int): The number of LSTM blocks to use in the generator.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, seq_length: int, num_lstm_blocks: int=1):
        super(Generator, self).__init__()
        self.seq_size = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.num_lstm_blocks = num_lstm_blocks
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_blocks, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ls = nn.LogSoftmax(dim=2)
    
    def forward(self, z, temp: int, hard: bool=False):
        """
        A forward pass through the generator. Generates a sequence of probabilities for each word in the vocabulary based on the input noise.

        Args:
            z (torch.Tensor): The input noise. Should be of shape [batch_size, seq_length, input_size]. Sampled from a uniform distribution.
            temp (int): The temperature parameter used to control the smoothness of the softmax function.
            hard (bool, optional): If True, the output will be one-hot encoded. Default is False.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, seq_length, output_size] containing the probabilities for each word in the vocabulary.
        """
        lstm_out, _ = self.lstm(z)
        linear = self.fc(lstm_out)
        logits = self.ls(linear)
        gumbel_softmaxed_logits = [gumbel_softmax(logits[:,i,:], temp, hard) for i in range(self.seq_size)]
        return torch.stack(gumbel_softmaxed_logits, dim=1)

class Discriminator(nn.Module):
    """
    Discriminator component of the GAN. Takes in a sequence of word embeddings and outputs a probability that the sequence is real.
    Embeddings are currently assumed to be of shape [batch_size, seq_length, input_dim] where input_dim is the dimension of the word embeddings.
    Word embeddings are currently represent as a one-hot vector or an approximation of a one-hot vector.

    Attributes:
        input_dim (int): The input dimension (number of features) for the discriminator.
        hidden_dim (int): The hidden dimension for the LSTM blocks.
        seq_length (int): The sequence length of the input sequences.
    """
    def __init__(self, input_dim, hidden_dim, seq_length):
        super(Discriminator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        A forward pass through the discriminator. Takes in a sequence of word embeddings and outputs a probability that the sequence is real.

        Args:
            x (torch.Tensor): The input sequence of word embeddings. Should be of shape [batch_size, seq_length, input_dim].

        Returns:
            torch.Tensor: A tensor of shape [batch_size, 1] containing the probabilities that the sequences are real.
        """
        # Assuming x is of shape [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        
        return torch.sigmoid(out)
    
def train(generator, discriminator, training_sentences, validation_sentences, word2vec_manager, calc_perplexity: Callable, seq_length, generator_input_features, generator_lr: float = 0.001, discriminator_lr: float = 0.001, num_epochs=2, batch_size=4, temperature=1.0, temp_decay_rate: float = 0.001, gumbel_hard: bool = False, encoding_method="one_hot", noise_sample_method: Literal['uniform', 'normal'] = 'uniform', device: str = 'cpu', tensorboard_log_dir: str = 'runs', deep_discriminator_metrics: bool = False): 
    """
    Trains a Generative Adversarial Network (GAN) consisting of a generator and discriminator.

    Args:
        generator (torch.nn.Module): The generator model.
        discriminator (torch.nn.Module): The discriminator model.
        training_sentences (List[List[str]]): Tokenized sentences for the dataset.
        validation_sentences (List[List[str]]): Tokenized sentences for the validation dataset.
        word2vec_manager (Any): Embedding manager to handle word embeddings.
        calc_perplexity (Callable): Function to calculate perplexity of the generator.
        seq_length (int): Sequence length for the sentences.
        generator_input_features (int): Input dimension (number of features) for the generator.
        generator_lr (float, optional): Learning rate for the generator optimizer. Default is 0.001.
        discriminator_lr (float, optional): Learning rate for the discriminator optimizer. Default is 0.001.
        num_epochs (int, optional): Number of epochs for training. Default is 2.
        batch_size (int, optional): Batch size for training. Default is 4.
        temperature (float, optional): Temperature parameter used in gumbel softmax equation. Default is 1.0.
        temp_decay_rate (float, optional): Rate at which the temperature parameter decays. Default is 0.001.
        gumbel_hard (bool, optional): If True, the output of the gumbel softmax function will be one-hot encoded. Default is False.
        encoding_method (str, optional): Encoding method for the data ('word_embedding' or 'one_hot'). Default is "one_hot".
        noise_sample_method (str, optional): Method for sampling noise for the generator ('uniform' or 'normal'). Default is 'uniform'.
        tensorboard_log_dir (str, optional): Directory to log tensorboard data. Default is 'runs'.
        deep_discriminator_metrics (bool, optional): If True, calculate additional metrics for the discriminator. Default is False.

    Returns:
        tuple: A tuple containing the average generator and discriminator losses for each epoch.
    """
    assert noise_sample_method in ['uniform', 'normal'], f'Invalid noise sample method: {noise_sample_method}'
    assert encoding_method in ['word_embedding', 'one_hot'], f'Invalid encoding method: {encoding_method}'

    writer = SummaryWriter(tensorboard_log_dir)

    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=generator_lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=discriminator_lr)

    # Initialize loss function
    loss = torch.nn.BCELoss()

    # Create data loader
    dataloader = create_embedding_dataloader(training_sentences, word2vec_manager, seq_length, batch_size, encoding_method, verbose=False)
    val_dataloader = create_embedding_dataloader(validation_sentences, word2vec_manager, seq_length, batch_size, encoding_method, verbose=False)
    avg_g_loss = 0
    avg_d_loss = 0

    bleu_smoothing = SmoothingFunction().method5

    for i in range(num_epochs):
        g_loss_total = 0
        d_loss_total = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {i+1}/{num_epochs}')
        val_dataset = iter(val_dataloader)

        # Train on batches 
        for batch_idx, real_data in enumerate(progress_bar):
            # Decay temperature if necessary
            if batch_idx % 100 == 1:
                temperature = max(temperature * np.exp(-temp_decay_rate * batch_idx), 0.5)

            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Generate fake data
            noise = generate_noise((batch_size, seq_length, generator_input_features), noise_sample_method).to(device)
            generated_data = generator(noise, temperature, hard=gumbel_hard)

            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            predictions_on_real_data = discriminator(real_data)
            real_loss = loss(predictions_on_real_data, real_labels)
            fake_labels = torch.zeros(batch_size, 1)
            fake_preds = discriminator(generated_data.detach())
            fake_loss = loss(fake_preds, fake_labels)
            d_loss = real_loss + fake_loss
            
            if deep_discriminator_metrics:
                # Metrics (accuracy, precision, recall, f1, loss)
                accuracy, precision, recall, f1 = get_discriminator_metrics(predictions_on_real_data, fake_preds) 
                writer.add_scalar('Discriminator Accuracy / Train', accuracy, batch_idx)
                writer.add_scalar('Discriminator Precision / Train', precision, batch_idx)
                writer.add_scalar('Discriminator Recall / Train', recall, batch_idx)
                writer.add_scalar('Discriminator F1 / Train', f1, batch_idx)
                writer.add_scalar('Discriminator Loss / Train', d_loss, batch_idx)
            
            # Backpropagate and update weights
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            noise = generate_noise((batch_size, seq_length, generator_input_features), noise_sample_method)
            generated_data = generator(noise, temperature, hard=gumbel_hard)  
            fake_pred = discriminator(generated_data)
            fake_labels = torch.ones(batch_size, 1)
            
            g_loss = loss(fake_pred, fake_labels)
            writer.add_scalar('Generator Loss / Train', g_loss, batch_idx)

            # Backpropagate and update weights
            g_loss.backward()
            optimizer_G.step()

            # Add metrics for both generator and discriminator loss on same graph
            writer.add_scalars('Loss / Train', {'Generator Loss': g_loss, 'Discriminator Loss': d_loss}, batch_idx)

            # Generator Metrics (besides loss)
            # Want to see how well the generator is tricking the discriminator
            gen_trick_accuracy = accuracy_score(fake_labels.detach().numpy(), fake_pred.detach().numpy() > 0.5)
            writer.add_scalar('Generator Trick Accuracy / Train', gen_trick_accuracy, batch_idx)

            if d_loss.item() < 0.2:
                print('Discriminator loss is too low, stopping training')
                break
            if batch_idx % 10 == 0:
                generated_sentences = turn_generator_output_to_text(generated_data, word2vec_manager)
                sentence_bleu_scores =[sentence_bleu(validation_sentences, generated_sentence, smoothing_function=bleu_smoothing) for generated_sentence in generated_sentences]
                mean_sentence_bleu_score = np.mean(sentence_bleu_scores)
                mean_perplexity = calc_perplexity(generated_sentences)
                writer.add_scalar('Sentence BLEU Score / Generated', mean_sentence_bleu_score, batch_idx)
                writer.add_scalar('Mean Perplexity / Generated', mean_perplexity, batch_idx)

                if mean_perplexity < 50:
                    print('Perplexity is too low, stopping training')
                    break

                if deep_discriminator_metrics:
                    # TODO: Calculate Validation Metrics (accuracy, precision, recall, f1, loss) for discriminator
                    # This will help us understand how well the discriminator is doing on the validation set (helps see if its overfitting)
                    val_data = next(val_dataset)
                    val_data = val_data.to(device)
                    val_preds = discriminator(val_data)
                    val_accuracy, val_precision, val_recall, val_f1 = get_discriminator_metrics(val_preds, fake_preds)
                    writer.add_scalar('Discriminator Accuracy / Validation', val_accuracy, batch_idx)
                    writer.add_scalar('Discriminator Precision / Validation', val_precision, batch_idx)
                    writer.add_scalar('Discriminator Recall / Validation', val_recall, batch_idx)
                    writer.add_scalar('Discriminator F1 / Validation', val_f1, batch_idx)
                    # loss 
                    val_loss = loss(val_preds, real_labels)
                    writer.add_scalar('Discriminator Loss / Validation', val_loss, batch_idx)


            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            num_batches += 1


            progress_bar.set_description(f'Epoch {i+1}/{num_epochs} | Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}')
        
    avg_g_loss = g_loss_total / num_batches
    avg_d_loss = d_loss_total / num_batches

    # Log hyperparameters and final average losses
    hparams = {
        'generator_lr': generator_lr,
        'discriminator_lr': discriminator_lr,
        'batch_size': batch_size,
        'temperature': temperature,
        # ... add other hyperparameters here
    }

    metrics = {
        'final_avg_generator_loss': avg_g_loss,
        'final_avg_discriminator_loss': avg_d_loss,
    }

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
    writer.flush()
    return avg_g_loss, avg_d_loss


def generate_noise(shape: tuple, noise_sample_method: Literal['uniform', 'normal'] = 'uniform'):
    """
    Generates noise for the generator.

    Args:
        shape (tuple): The shape of the noise tensor.
        noise_sample_method (str, optional): Method for sampling noise for the generator ('uniform' or 'normal'). Default is 'uniform'.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, seq_length, input_size] containing the noise for the generator.
    """
    assert noise_sample_method in ['uniform', 'normal'], f'Invalid noise sample method: {noise_sample_method}'
    if noise_sample_method == 'uniform':
        return torch.rand(shape)
    else:
        return torch.randn(shape)

def get_discriminator_metrics(real_data_predictions: torch.Tensor, generated_data_predictions: torch.Tensor):
    """
    Calculates the accuracy, precision, recall, and f1 score for the discriminator. Assuming discriminator is binary classifier (has sigmoid as final layer).

    Args:
        real_data_predictions (torch.Tensor): The predictions for the real data.
        generated_data_predictions (torch.Tensor): The predictions for the generated data.
    
    Returns:
        tuple: A tuple containing the accuracy, precision, recall, and f1 score for the discriminator.
    """
    real_labels = torch.ones(real_data_predictions.shape[0], 1)
    fake_labels = torch.zeros(generated_data_predictions.shape[0], 1)
    labels = torch.cat((real_labels, fake_labels)).flatten().numpy().astype(int)
    predictions = torch.cat((real_data_predictions, generated_data_predictions)).detach().flatten().numpy() > 0.5
    predictions = predictions.astype(int)
    accuracy = accuracy_score(labels, predictions)
    percision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, percision, recall, f1

def turn_generator_output_to_text(gen_out: torch.Tensor, word2vec_manager: Any):
    """
    Turns the generator output into text.

    Args:
        gen_out (torch.Tensor): The generator output. Should be of shape [batch_size, seq_length, output_size].
        word2vec_manager (Any): Embedding manager to handle word embeddings.

    Returns:
        List[str]: A list of strings containing the generated text.
    """
    # need to argmax final dimension of gen_out
    # then use word2vec_manager to turn the indices into words
    # then return the list of words
    batch_size, seq_length, output_size = gen_out.shape
    gen_out = gen_out.argmax(dim=2)
    results = []
    for i in range(batch_size):
        sentence = []
        for index in gen_out[i]:
            sentence.append(word2vec_manager.index_to_word(index))
        results.append(sentence)
    return results