from mindiff import nn


class MLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, X):
        l1 = self.linear_1(X)
        r1 = self.relu_1(l1)
        l2 = self.linear_2(r1)
        out = self.softmax(l2)
        return out
