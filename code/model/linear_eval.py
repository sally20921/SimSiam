import torch.nn as nn
import torchvision
import torch

class MLP(nn.Module):
    def __init__(self, args, pt_model, num_classes)
        super(MLP, self).__init__()
        n_channels = pt_model.output_dim
        self.clasifier = nn.Sequential()
        self.classifier.add_module('W1', nn.Linear(n_channels, num_classes))

    def forward(self, x):
        return self.classifier(x)

class LinearEvalModel(nn.Module):

    def __init__(self, args, pt_model, num_classes):
        super(LinearEvalModel, self).__init__()

        self.encoder = pt_model.encoder
        self.mlp = MLP(args, pt_model, num_clases)


    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        output = self.mlp(h)
        return output
