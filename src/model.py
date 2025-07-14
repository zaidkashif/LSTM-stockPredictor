import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
    
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out,_=self.lstm(x)
        last_out=lstm_out[:,-1,:]
        output=self.fc(last_out)
        return output
    