import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
from embeddings import create_embedding_dataloader

def gumbel_softmax(logits, temperature: int):
    """
    Implements the gumbel softmax function described in https://arxiv.org/pdf/1611.04051.pdf

    Args:
        logits (torch.Tensor): The logits to apply the gumbel softmax function to.
        temperature (int): The temperature parameter used to control the smoothness of the softmax function.
    """
    U = torch.rand(logits.shape)
    G = -torch.log(-torch.log(U))
    return F.softmax((logits + G) / temperature, dim=1)

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
    
    def forward(self, z, temp: int, should_sample: bool=False):
        """
        A forward pass through the generator. Generates a sequence of probabilities for each word in the vocabulary based on the input noise.

        Args:
            z (torch.Tensor): The input noise. Should be of shape [batch_size, seq_length, input_size]. Sampled from a uniform distribution.
            temp (int): The temperature parameter used to control the smoothness of the softmax function.
            should_sample (bool, optional): If True, the output will be sampled from the gumbel softmax distribution. Default is False.

        Returns:
            if should_sample is True, returns a tensor of shape [batch_size, seq_length] containing a sequence sampled words indices.
            if should_sample is False, returns a tensor of shape [batch_size, seq_length, output_size] containing the probabilities for each word in the vocabulary.
        """
        lstm_out, _ = self.lstm(z)
        logits = self.fc(lstm_out)
        gumbel_softmaxed_logits = [gumbel_softmax(logits[:,i,:], temp) for i in range(self.seq_size)]
        if should_sample:
            sampled_words = [torch.multinomial(gumbel_softmaxed_logits[i], 1)[0] for i in range(self.seq_size)]
            return torch.stack(sampled_words, dim=1).squeeze()
        else:
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
    
def train(generator, discriminator, tokenized_sentences, word2vec_manager, seq_length, generator_input_features, num_epochs=2, batch_size=4, learning_rate=0.001, temperature=1.0, encoding_method="one_hot", debug=True):
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
        encoding_method (str, optional): Encoding method for the data ('word_embedding' or 'one_hot'). Default is "one_hot".
        debug (bool, optional): If True, debug information will be printed. Default is True.

    Returns:
        tuple: A tuple containing the average generator and discriminator losses for each epoch.
    """

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Initialize loss function
    loss = torch.nn.BCELoss()

    # Create data loader
    dataloader = create_embedding_dataloader(tokenized_sentences, word2vec_manager, seq_length, batch_size, encoding_method, verbose=debug)
    loss_history = []

    for i in range(num_epochs):
        epoch_g_loss = 0.0  # To accumulate generator loss over the epoch
        epoch_d_loss = 0.0  # To accumulate discriminator loss over the epoch

        # Train on batches 
        for real_data in tqdm(dataloader, desc=f'Epoch {i+1}/{num_epochs}'):
            batch_size = real_data.size(0)

            # Generate fake data
            noise = torch.randn(batch_size, seq_length, generator_input_features)
            generated_data = generator(noise, temperature)

            # Train discriminator on real and fake data
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            real_loss = loss(discriminator(real_data), real_labels)
            fake_labels = torch.zeros(batch_size, 1)
            fake_loss = loss(discriminator(generated_data.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            generated_data = generator(noise, temperature)  
            fake_pred = discriminator(generated_data)
            g_loss = loss(fake_pred, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        if debug:
            print(f'Epoch {i+1}/{num_epochs} | Generator loss: {avg_g_loss} | Discriminator loss: {avg_d_loss}')
        
        loss_history.append((avg_g_loss, avg_d_loss))

    return zip(*loss_history)
