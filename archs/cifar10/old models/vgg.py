"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
# change num of classes here
    def __init__(self, features, num_class=5):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))



# an easier version of the same code with manually added layers
# class VGG16(torch.nn.Module):
# def __init__(self, n_classes):
#     super(VGG16, self).__init__()

#     # construct model
#     self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
#     self.conv11_bn = nn.BatchNorm2d(64)
#     self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
#     self.conv12_bn = nn.BatchNorm2d(64)
#     self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#     self.conv21_bn = nn.BatchNorm2d(128)
#     self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
#     self.conv22_bn = nn.BatchNorm2d(128)
#     self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
#     self.conv31_bn = nn.BatchNorm2d(256)
#     self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
#     self.conv32_bn = nn.BatchNorm2d(256)
#     self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
#     self.conv33_bn = nn.BatchNorm2d(256)
#     self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
#     self.conv41_bn = nn.BatchNorm2d(512)
#     self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
#     self.conv42_bn = nn.BatchNorm2d(512)
#     self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
#     self.conv43_bn = nn.BatchNorm2d(512)
#     self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#     self.conv51_bn = nn.BatchNorm2d(512)        
#     self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#     self.conv52_bn = nn.BatchNorm2d(512)
#     self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
#     self.conv53_bn = nn.BatchNorm2d(512)

#     self.fc6 = nn.Linear(512, 512)
#     self.fc7 = nn.Linear(512, 512)
#     self.fc8 = nn.Linear(512, n_classes)

# def forward(self, x):
#     x = F.relu(self.conv11_bn(self.conv1_1(x)))
#     x = F.relu(self.conv12_bn(self.conv1_2(x)))
#     x = F.max_pool2d(x, (2, 2))

#     x = F.relu(self.conv22_bn(self.conv2_1(x)))
#     x = F.relu(self.conv21_bn(self.conv2_2(x)))
#     x = F.max_pool2d(x, (2, 2))

#     x = F.relu(self.conv31_bn(self.conv3_1(x)))
#     x = F.relu(self.conv32_bn(self.conv3_2(x)))
#     x = F.relu(self.conv33_bn(self.conv3_3(x)))
#     x = F.max_pool2d(x, (2, 2))

#     x = F.relu(self.conv41_bn(self.conv4_1(x)))
#     x = F.relu(self.conv42_bn(self.conv4_2(x)))
#     x = F.relu(self.conv43_bn(self.conv4_3(x)))
#     x = F.max_pool2d(x, (2, 2))

#     x = F.relu(self.conv51_bn(self.conv5_1(x)))
#     x = F.relu(self.conv52_bn(self.conv5_2(x)))
#     x = F.relu(self.conv53_bn(self.conv5_3(x)))
#     x = F.max_pool2d(x, (2, 2))

#     x = x.view(-1, self.num_flat_features(x))

#     x = F.relu(self.fc6(x))
#     x = F.relu(self.fc7(x))
#     x = self.fc8(x)
#     return x