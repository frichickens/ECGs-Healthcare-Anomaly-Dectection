import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super().__init__()
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.rnn1(x)               # out: (batch, seq_len, hidden_dim)
        _, (h_n, _) = self.rnn2(out)        # h_n: (1, batch, embedding_dim)
        z = h_n.squeeze(0)                  # z: (batch, embedding_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=32, n_features=1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, z):
        # z: (batch, embedding_dim)
        batch_size = z.size(0)
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, embedding_dim)

        out, _ = self.rnn1(z_seq)
        out, _ = self.rnn2(out)

        out_flat = out.contiguous().view(-1, self.hidden_dim)
        x_hat_flat = self.output_layer(out_flat)
        x_hat = x_hat_flat.view(batch_size, self.seq_len, -1)  # (batch, seq_len, n_features)
        return x_hat


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super().__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
