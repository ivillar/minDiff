from mindiff import nn


class OldMLPClassifier(nn.Module):
    def __init__(self, layer_size_list):
        super().__init__()
        self.layers = []
        for i in range(len(layer_size_list) - 2):
            old_dims = layer_size_list[i]
            new_dims = layer_size_list[i + 1]
            self.layers.append(nn.Linear(old_dims, new_dims))
            self.layers.append(nn.ReLU())
        old_dims = layer_size_list[-2]
        new_dims = layer_size_list[-1]
        self.layers.append(nn.Linear(old_dims, new_dims))
        self.layers.append(nn.Softmax())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x


class NewMLPClassifier(nn.Module):
    def __init__(self, layer_size_list):
        super().__init__()
        self.layers = []
        for i in range(len(layer_size_list) - 2):
            old_dims = layer_size_list[i]
            new_dims = layer_size_list[i + 1]
            self.layers.append(nn.Linear(old_dims, new_dims))
            self.layers.append(nn.ReLU())
        old_dims = layer_size_list[-2]
        new_dims = layer_size_list[-1]
        self.layers.append(nn.Linear(old_dims, new_dims))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x
