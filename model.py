import torch
import torch.nn as nn
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = resnet18(pretrained=True)   # 학습된 모델
        self.head = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.head(x)


if __name__ == '__main__':
    import numpy as np
    model = Model()

    x = np.zeros((1,3,128,128))  # batch, channel, height, width
    x = torch.tensor(x).float()

    y = model.forward(x)

    print(y.size())