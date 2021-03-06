import torch
from torch.nn import Module
import torch.nn.functional as F


class VGGFeature(Module):
    def __init__(self):
        super(VGGFeature, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        relu1_1  = F.relu(self.conv1_1(x))
        relu1_2  = F.relu(self.conv1_2(relu1_1))
        maxpool_1  = F.max_pool2d(relu1_2, kernel_size=2, stride=2)
        relu2_1  = F.relu(self.conv2_1(maxpool_1))
        relu2_2  = F.relu(self.conv2_2(relu2_1))
        maxpool_2  = F.max_pool2d(relu2_2, kernel_size=2, stride=2)
        relu3_1  = F.relu(self.conv3_1(maxpool_2))
        relu3_2  = F.relu(self.conv3_2(relu3_1))
        relu3_3  = F.relu(self.conv3_3(relu3_2))
        maxpool_3  = F.max_pool2d(relu3_3, kernel_size=2, stride=2)
        relu4_1  = F.relu(self.conv4_1(maxpool_3))
        relu4_2  = F.relu(self.conv4_2(relu4_1))
        relu4_3  = F.relu(self.conv4_3(relu4_2))

        return [relu1_2,relu2_2,relu3_3,relu4_3]

class ResidualBlock(Module):
    def __init__(self,num):
        super(ResidualBlock, self).__init__()
        self.c1 = torch.nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.c2 = torch.nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.b1 = torch.nn.BatchNorm2d(num)
        self.b2 = torch.nn.BatchNorm2d(num)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x

class ImageTransformNet(Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        self.c1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.c2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.c3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(128)
        self.r2 = ResidualBlock(128)
        self.r3 = ResidualBlock(128)
        self.r4 = ResidualBlock(128)
        self.r5 = ResidualBlock(128)
        self.d1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.d2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.d3 = torch.nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.b1 = torch.nn.BatchNorm2d(32)
        self.b2 = torch.nn.BatchNorm2d(64)
        self.b3 = torch.nn.BatchNorm2d(128)
        self.b4 = torch.nn.BatchNorm2d(64)
        self.b5 = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = F.relu(self.b4(self.d1(h)))
        h = F.relu(self.b5(self.d2(h)))
        y = self.d3(h)
        return y
