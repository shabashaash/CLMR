import torch.nn as nn

class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x