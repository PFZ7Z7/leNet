import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset, DataLoader

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  #(input-0.5)/0.5

        # 50000张训练图片
        # 第一次使用时要将download设置为True才会自动下载数据集
    train_set = torchvision.datasets.CIFAR10(root = './data', train = True,
                                             download = False, transform = transform)
    # torchvision.datasets.(有很多数据集可以自行下载使用)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 36,
                                               shuffle = True, num_workers = 0) #线程数windows0 linux可以自己设置


#     # 10000张验证图片
#     # 第一次使用时要将download设置为True才会自动下载数据集
    val_set = torchvision.datasets.CIFAR10(root = './data', train = False,
                                           download = False, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 10000,
                                             shuffle = False, num_workers = 0)
    #
    val_data_iter = iter(val_loader)     #转化为一个可以迭代的迭代器
    val_image, val_label = val_data_iter.next()
    #
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'truck')
# def imshow(img):
#     img = img / 2 + 0.5   #unnormalize  和上边的标准化反过来
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
# print(' '.join('%5s' %classes[val_label[j]] for j in range(5)))
# imshow(torchvision.utils.make_grid(val_image))

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    for epoch in range(10):

        running_loss = 0.0
        for step, data in enumerate(train_loader, start = 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()              #历史损失梯度清零，每计算一个batch就要调用一次
            # forward + backward + optimize
            outputs = net(inputs)               #先正向传播得到输出
            loss = loss_function(outputs, labels)
            loss.backward()                     #反向传播
            optimizer.step()                    #参数更新

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():           #with是一个上下文管理器，这里表示不计算梯度了，不然会爆内存
                    outputs = net(val_image) # [batch, 10]
                    predict_y = torch.max(outputs, dim = 1)[1]   #最可能是那个类的的index
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)  #得到本次测试中预测对了多少个样本。
                                                                #tensor.item()取出这个数值
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step +1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')

    save_path = './lenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()