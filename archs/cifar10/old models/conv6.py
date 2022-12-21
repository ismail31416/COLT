import torch
import torch.nn as nn
import torch.nn.functional as F
 
class conv6(nn.Module):
    def __init__(self, num_classes=5):
        super(conv6, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # input [20, 1, 32, 32]
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1) 
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1) 
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1) 
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1) 
        self.pool3 = nn.MaxPool2d(2,2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*6*6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0),-1)
        # x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output