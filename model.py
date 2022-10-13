import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):                         #继承父类
    def __init__(self):
        super(LeNet, self).__init__()           #定义网络层结构
        self.conv1 = nn.Conv2d(3, 16, 5)        #输入特征层的深度彩色图片通道3，16个卷积核，kernel尺寸5*5
        self.pool1 = nn.MaxPool2d(2, 2)         ##下采样层 （池化核2*2，步距2*2）
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)       #全连接曾需要的是一个一维向量，需要将32*5*5展平
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)            #最后分为10类

    def forward(self, x):    #正向传播的过程  #N=(W-F+2P)/S+1   (32-5+0)/1+1=28
        x = F.relu(self.conv1(x))   # input(3, 32, 32)  output(16, 28, 28)
        x = self.pool1(x)           # output(16, 14, 14)   #下采样层
        x = F.relu(self.conv2(x))   # output(32, 10, 10)
        x = self.pool2(x)           # output(32, 5, 5)
        x = x.view(-1, 32*5*5)      # output(32*5*5)
        x = F.relu(self.fc1(x))     # output(120)
        x = F.relu(self.fc2(x))     # output(84)
        x = self.fc3(x)             # output(10)
        return x


