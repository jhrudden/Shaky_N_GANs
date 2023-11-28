import torch
from torch import nn
from torch.nn import functional as F

def gumbel_softmax(logits, temperature: int):
    U = torch.rand(logits.shape)
    G = -torch.log(-torch.log(U))
    return F.softmax((logits + G) / temperature, dim=1)

class Generator(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, seq_length: int, num_lstm_blocks: int=1):
        super(Generator, self).__init__()
        self.seq_size = seq_length
        self.input_size = input_size
        self.output_size = output_size
        self.num_lstm_blocks = num_lstm_blocks
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_blocks, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, z, temp: int):
        lstm_out, _ = self.lstm(z)
        logits = self.fc(lstm_out)
        return torch.stack([gumbel_softmax(logits[:,i,:], temp) for i in range(self.seq_size)])

# TODO: question: What shape is the input to the discriminator? Should it be a sequence of embeddings or a sequence of one-hot vectors?
# TODO: Do I need to sample from the generator and then feed the output to the discriminator?
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length):
        super(Discriminator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        
        return torch.sigmoid(out)