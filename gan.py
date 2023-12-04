from typing import Literal
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from embeddings import create_embedding_dataloader
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

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
    
def train(generator, discriminator, tokenized_sentences, word2vec_manager, seq_length, generator_input_features, num_epochs=2, batch_size=4, learning_rate=0.001, temperature=1.0, temp_decay_rate: float = 0.001, gumbel_hard: bool = False, encoding_method="one_hot", noise_sample_method: Literal['uniform', 'normal'] = 'uniform', device: str = 'cpu', debug=True):
    """
    Trains a Generative Adversarial Network (GAN) consisting of a generator and discriminator.

    Args:
        generator (torch.nn.Module): The generator model.
        discriminator (torch.nn.Module): The discriminator model.
        tokenized_sentences (List[List[str]]): Tokenized sentences for the dataset.
        word2vec_manager (Any): Embedding manager to handle word embeddings.
        seq_length (int): Sequence length for the sentences.
        generator_input_features (int): Input dimension (number of features) for the generator.
        num_epochs (int, optional): Number of epochs for training. Default is 2.
        batch_size (int, optional): Batch size for training. Default is 4.
        learning_rate (float, optional): Learning rate for the optimizers. Default is 0.001.
        temperature (float, optional): Temperature parameter used in gumbel softmax equation. Default is 1.0.
        temp_decay_rate (float, optional): Rate at which the temperature parameter decays. Default is 0.001.
        gumbel_hard (bool, optional): If True, the output of the gumbel softmax function will be one-hot encoded. Default is False.
        encoding_method (str, optional): Encoding method for the data ('word_embedding' or 'one_hot'). Default is "one_hot".
        noise_sample_method (str, optional): Method for sampling noise for the generator ('uniform' or 'normal'). Default is 'uniform'.
        debug (bool, optional): If True, debug information will be printed. Default is True.

    Returns:
        tuple: A tuple containing the average generator and discriminator losses for each epoch.
    """
    assert noise_sample_method in ['uniform', 'normal'], f'Invalid noise sample method: {noise_sample_method}'
    assert encoding_method in ['word_embedding', 'one_hot'], f'Invalid encoding method: {encoding_method}'

    writer = SummaryWriter()

    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Initialize loss function
    loss = torch.nn.BCELoss()

    # Create data loader
    dataloader = create_embedding_dataloader(tokenized_sentences, word2vec_manager, seq_length, batch_size, encoding_method, verbose=debug)
    avg_g_loss = 0
    avg_d_loss = 0

    for i in range(num_epochs):
        g_loss_total = 0
        d_loss_total = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {i+1}/{num_epochs}')

        # Train on batches 
        for batch_idx, real_data in enumerate(progress_bar):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train discriminator on real and fake data

            # Generate fake data
            noise = generate_noise((batch_size, seq_length, generator_input_features), noise_sample_method).to(device)
            generated_data = generator(noise, temperature, hard=gumbel_hard)

            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            real_loss = loss(discriminator(real_data), real_labels)
            fake_labels = torch.zeros(batch_size, 1)
            fake_preds = discriminator(generated_data.detach())
            fake_loss = loss(fake_preds, fake_labels)
            d_loss = real_loss + fake_loss
            
            # Metrics
            accuracy = accuracy_score(torch.cat((real_labels, fake_labels)).detach().numpy(), torch.cat((discriminator(real_data), fake_preds)).detach().numpy() > 0.5)
            writer.add_scalar('Discriminator Accuracy / Train', accuracy, batch_idx)
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
            
            # metrics
            g_loss = loss(fake_pred, fake_labels)
            accuracy = accuracy_score(fake_labels.detach().numpy(), fake_pred.detach().numpy() > 0.5)
            writer.add_scalar('Generator Trick Accuracy / Train', accuracy, batch_idx)
            writer.add_scalar('Generator Loss / Train', g_loss, batch_idx)

            # Backpropagate and update weights
            g_loss.backward()
            optimizer_G.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            num_batches += 1

            progress_bar.set_description(f'Epoch {i+1}/{num_epochs} | Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}')

            # Decay temperature
            if batch_idx % 100 == 0:
                temperature = max(temperature * np.exp(-temp_decay_rate * batch_idx), 0.5)

        if debug:
            print(f'Epoch {i+1}/{num_epochs} | Generator loss: {avg_g_loss} | Discriminator loss: {avg_d_loss}')
        
    avg_g_loss = g_loss_total / num_batches
    avg_d_loss = d_loss_total / num_batches

    # Log hyperparameters and final average losses
    hparams = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'temperature': temperature,
        'temp_decay_rate': temp_decay_rate,
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
    