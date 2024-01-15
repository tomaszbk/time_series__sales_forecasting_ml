from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout= 0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        # extract only the last time step
        # x = x[:, -1]
        x = self.linear(x)
        return x

# Configuración del modelo
input_size = 40
hidden_size = 64
num_layers = 2
output_size = 1
dropout = 0.1

# Crear una instancia del modelo
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)