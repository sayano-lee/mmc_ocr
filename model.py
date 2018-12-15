import torch.nn as nn
from torchvision import models


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AvgPool2d(kernel_size=16)

    def forward(self, input):
        x = self.model(input)
        x = self.pool(x)
        return x.squeeze()


if __name__ == '__main__':
    pass
