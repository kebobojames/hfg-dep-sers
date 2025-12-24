import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class MLP(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate = 0):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Sequential(
            nn.Linear(input_shape, 2048),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        out = self.fc1(x)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.fc2(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    dnn = MLP(882, 0.1)
    print(dnn)
    y= dnn(torch.randn(1,1,882))