import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,     # Each timestep has 1 feature (closing price)
            hidden_size=hidden_size,   # Number of units in hidden state
            num_layers=num_layers,     # Number of LSTM layers stacked
            batch_first=True           # Input shape: (batch, seq, feature)
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)          # lstm_out: (batch, seq_len, hidden)
        last_out = lstm_out[:, -1, :]       # Take the output at the last timestep
        output = self.fc(last_out)          # Final prediction
        return output
