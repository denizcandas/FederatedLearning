# https://nextjournal.com/gkoehler/pytorch-mnist

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision


batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
local_epochs = 10

train_losses = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(network, data, target, fed_sgd=True):
    iterations = 1
    l_epochs = 1
    if not fed_sgd:
        data = torch.split(data, 64)  # split into 64 part batches if using fed_avg
        target = torch.split(target, 64)
        l_epochs = local_epochs
    else:
        data = [data]  # wrap data in an array so that data has always the same nesting count
        target = [target]

    for local_epoch in range(l_epochs):
        for i in range(iterations):
            optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                  momentum=momentum)
            network.train()
            optimizer.zero_grad()
            output = network(data[i])
            loss = F.nll_loss(output, target[i])
            loss.backward()
            if not fed_sgd:
                optimizer.step()
                optimizer.zero_grad()
        train_losses.append(loss)
        print(f'Local epoch: {local_epoch}, Batch loss: {loss}')
    if fed_sgd:
        # https://discuss.pytorch.org/t/please-help-how-can-copy-the-gradient-from-net-a-to-net-b/41226/5
        return [param[1].grad for param in network.named_parameters()]
    else:
        return network.state_dict()


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


def test(network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
