import torch.nn as nn
import torch.nn.functional as F

class CNN_model(nn.Module):
   def __init__(self,num_classes=12):
        super().__init__()
        #Convolution layer1, input 224*224*3, compute (224 + 2*3 - (7-1)-1)/2 +1 = 116
        self.conv1 = nn.Conv2d(4,64, kernel_size=7, stride=2, padding=3, bias=False)
        #pooling layer, input 118*118*64, compute 116/2 = 58
        self.pool1 = nn.MaxPool2d(2,2)

        #Convolution layer2, input 58*58*64, compute (58 + 2*2 - (5-1)-1)/2 +1 = 29
        self.conv2 = nn.Conv2d(64,128, kernel_size=5, stride=2, padding=2, bias=False)
        #pooling layer, input 29*29*128, compute 29/2 = 14
        self.pool2 = nn.MaxPool2d(2,2)

        #Convolution layer2, input 14*14*128, compute (14 + 2*2 - (3-1)-1)/2 +1 = 8
        self.conv3 = nn.Conv2d(128,64, kernel_size=3, stride=2, padding=2, bias=False)
        #pooling layer, input 8*8*64, compute 8/2 = 4
        self.pool3 = nn.MaxPool2d(2,2)

        #fully connected network
        #input 4*4*64
        self.fc1 = nn.Linear(4*4*64,128)
        self.fc2 = nn.Linear(128,num_classes)


   def forward(self, x):
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x1)))
        x3 = self.pool3(F.relu(self.conv3(x2)))

        x4 = x3.view(-1,4*4*64)
        x5 = F.relu(self.fc1(x4))
        x6 = self.fc2(x5)
        return x6
