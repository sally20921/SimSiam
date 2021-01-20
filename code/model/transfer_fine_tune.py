import torch.nn as nn
import torchvision
import torch

class FineTuningModel(nn.Module):

    def __init__(self, args, pt_model, num_classes):
        super(FineTuningModel, self).__init__()
        self.model = torchvision.models.resnet50()
        num_ftrs = self.model.fc.in_features
        # fc layer now matches the number of classes
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        # fc layer in the encoder now matches the number of classes
        pt_model.encoder.fc = torch.nn.Linear(num_ftrs, num_classes)

        state_dict = pt_model.encoder.state_dict()
        # load state dict into classification model
        self.model.load_state_dict(state_dict)

    def __call__(self):
        return self.model
