from torch import nn
import torch
import torchvision
from torchvision import models


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(
            config.Classifier.in_features_dim, config.Classifier.num_classes
        )

    def forward(self, x):
        return self.classifier(x)
