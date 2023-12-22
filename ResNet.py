import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, num_classes):
        '''
        This is a resnet50 model with 1 dense layer.
        ResNet50 is a pretrained model on ImageNet.
        '''
        super(ResNet, self).__init__()
        self.model = resnet50(pretrained=True) #resnet50
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
      
    def getname(self):
        '''
        Model name: ResNet50
        '''
        return 'ResNet50'
