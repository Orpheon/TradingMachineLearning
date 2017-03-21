import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

import readdata

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, receptivefield):
        self.data = data
        self.receptivefield = receptivefield

    def __len__(self):
        return len(self.data) - self.receptivefield

    def __getitem__(self, idx):
        if self.data[idx] > self.data[idx-1]:
            target = torch.Tensor([1])
        else:
            target = torch.Tensor([0])
        return torch.Tensor(self.data[idx-self.receptivefield:idx]), target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layerexp = 2
        self.conv = nn.ModuleList()
        for n in range(self.layerexp-1, -1, -1):
            self.conv.append(nn.Conv1d(1, 1, 5, padding=1, dilation=2**n).cuda())

    def forward(self, x):
        for conv in self.conv:
            print(x.size())
            x = nn.functional.relu(conv(x))
        return min(max(x, 0), 1)

    def receptivefield(self):
        return 2**(self.layerexp+1) - 1

    def build_datatensor(self, data):
        l = []
        t = []
        for idx in range(self.receptivefield(), len(data)):
            l.append(data[idx-self.receptivefield():idx-1][::-1])
            t.append(data[idx] > data[idx-1])
        return torch.Tensor(l), torch.Tensor(t)


print("Creating network..")
net = Net()

print("Loading data from disk..")
traindata, testdata = readdata.load("itbitUSD")

print("Building training dataset..")
trainset = Dataset(traindata, net.receptivefield())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=True, num_workers=2)

print("Building testing dataset tensor..")
testset = Dataset(testdata, net.receptivefield())
testloader = torch.utils.data.DataLoader(testset, batch_size=80, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

print("Starting main loop..")

for epoch in range(5):
  running_loss = 0
  for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.data[0]
    if i % 1000 == 999:
      print("[{0}, {1}] loss: {2}".format(epoch+1, i+1, running_loss/1000))
      running_loss = 0
print("Done")


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
