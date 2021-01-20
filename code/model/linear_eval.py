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

class LinearLayer(nn.Module):

    def __init__(self, args, pt_model, num_classes):
        super(LinearLayer, self).__init__()

        self.encoder = pt_model.encoder
        self.mlp = MLP(args, pt_model, num_clases)


    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        output = self.mlp(h)
        return output

class SupervisedHead(nn.Module):
    def __init__(self, args, pt_model, is_training, num_classes):
        super(SupervisedHead, self).__init__()
        self.linear_layer = LinearLayer(args, pt_model, num_classes)

    def forward(x):
        self.linear_layer(x)

