import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Conv(nn.Module):
    def __init__(self, num_classes):
        '''
        This is a simple CNN model with 3 conv layers and 2 dense layers.
        '''
        super(Conv, self).__init__()
        
        # conv layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # pool layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # dense layers
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 256 * 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def getname(self):
        '''
        Model name: Conv
        '''
        return 'Conv'
