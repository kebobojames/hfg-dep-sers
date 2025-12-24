import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ThreeCNN(nn.Module):
    def __init__(self, input_dim=1030, hidden_size=64, n_classes=3):
        super(ThreeCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=21, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        feature_size = self._get_encoding_size()
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh()
        )

        self.pool_dense = nn.Sequential(
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(p=0.5)
        )

        self.final = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.BatchNorm1d(n_classes)
        )

        self._initialize_weights()

    def encode(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.encode(x)
        out = self.classifier(out)
        out = out.view(out.size(0), 1, -1)
        out = self.pool_dense(out)
        out = out.view(out.size(0), -1)
        out = self.final(out)
        return out

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(10, 1, self.input_dim))
        z = self.encode(temp)
        return z.shape[-1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.05 ** 0.5)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.05 ** 0.5)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05 ** 0.5)
                nn.init.normal_(m.bias, mean=0.0, std=0.05 ** 0.5)

if __name__ == "__main__":
    cnn = ThreeCNN(input_dim=1030)
    print(cnn)
    y= cnn(torch.randn(10,1,1030))