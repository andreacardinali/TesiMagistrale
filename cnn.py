import torch
from torch import nn
import torch.nn.functional as F

class CNN1(nn.Module):
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):
    # https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
	# Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class CNN3(nn.Module):
    # https://studentsxstudents.com/training-a-convolutional-neural-network-cnn-on-cifar-10-dataset-cde439b67bf3
    def __init__(self, num_classes):
        super().__init__()
        def ConvLayer(inp, out, ks=3, s=1, p=1):
            return nn.Conv2d(inp, out, kernel_size=ks, stride=s, padding=p)
        self.neural_net = nn.Sequential(
            ConvLayer(3, 32), nn.ReLU(),
            ConvLayer(32, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            ConvLayer(64, 128), nn.ReLU(),
            ConvLayer(128, 256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            ConvLayer(256, 512), nn.ReLU(),
            ConvLayer(512, 1024), nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Flatten(),
            nn.Linear(1024*4*4, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.neural_net(x)

class CNN4(nn.Module):
    # https://github.com/cybertronai/pytorch-sso/blob/master/examples/distributed/classification/models/vgg.py
    # VGG19 variant
    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)), inplace=True)
        h = F.relu(self.bn1_2(self.conv1_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn2_1(self.conv2_1(h)), inplace=True)
        h = F.relu(self.bn2_2(self.conv2_2(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn3_1(self.conv3_1(h)), inplace=True)
        h = F.relu(self.bn3_2(self.conv3_2(h)), inplace=True)
        h = F.relu(self.bn3_3(self.conv3_3(h)), inplace=True)
        h = F.relu(self.bn3_4(self.conv3_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn4_1(self.conv4_1(h)), inplace=True)
        h = F.relu(self.bn4_2(self.conv4_2(h)), inplace=True)
        h = F.relu(self.bn4_3(self.conv4_3(h)), inplace=True)
        h = F.relu(self.bn4_4(self.conv4_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.bn5_1(self.conv5_1(h)), inplace=True)
        h = F.relu(self.bn5_2(self.conv5_2(h)), inplace=True)
        h = F.relu(self.bn5_3(self.conv5_3(h)), inplace=True)
        h = F.relu(self.bn5_4(self.conv5_4(h)), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out
