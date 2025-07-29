import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], activation=F.relu, dropout=0.2, device=None):
        super(ModularNN, self).__init__()

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.model = nn.Sequential(*layers).to(self.device)

        self.init_weights()

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.to(self.device)

        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
                x = self.dropout(x)
            else:
                x = layer(x)

        x = self.model[-1](x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output


# Exemple d'utilisation professionnel
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 100
    output_dim = 10
    hidden_dims = [256, 128, 64]

    model = ModularNN(input_dim=input_dim,
                      output_dim=output_dim,
                      hidden_dims=hidden_dims,
                      activation=F.relu,
                      dropout=0.3,
                      device=device)

    print(model)

    # Exemple d'input
    example_input = torch.randn(32, input_dim).to(device)

    # Forward pass
    output = model(example_input)
    print(output.shape)