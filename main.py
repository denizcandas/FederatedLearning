import torch
import torchvision
import random
import mnist_tutorial
import matplotlib.pyplot as plt


def experiment(user_count, user_fraction, c_epochs, c_rate, fed_sgd=True):
    active_user_count = int(user_count * user_fraction)
    network_list = []
    user_data = []
    user_target = []

    cn = mnist_tutorial.Net()  # central_network
    for user in range(user_count):
        network_list.append(mnist_tutorial.Net())

    for batch_idx, (data, target) in enumerate(train_loader(user_count)):
        print('Preparing data for user', batch_idx)
        user_data.append(data)
        user_target.append(target)
    print('Data loaded')

    optimizer = torch.optim.SGD(cn.parameters(), lr=c_rate, momentum=0.5)

    test_losses = [mnist_tutorial.test(cn)]

    for c_epoch in range(c_epochs):
        print(f'GLOBAL EPOCH {c_epoch}')
        chosen_users = random.sample(range(user_count), active_user_count)
        if fed_sgd:
            for user_number in chosen_users:
                user_net = network_list[user_number]
                user_net.load_state_dict(cn.state_dict())
                print(f'Local user {user_number}: ', end='')
                for cn_param, user_gradient in zip(cn.named_parameters(),
                                                   mnist_tutorial.train(user_net, user_data[user_number],
                                                                        user_target[user_number])):
                    if cn_param[1].grad is not None:
                        cn_param[1].grad += user_gradient.clone() / active_user_count
                    else:
                        cn_param[1].grad = user_gradient.clone() / active_user_count
            optimizer.step()
            optimizer.zero_grad()
        else:
            weights = []
            for user_number in chosen_users:
                user_net = network_list[user_number]
                user_net.load_state_dict(cn.state_dict())
                print(f'Local user {user_number}: ', end='')
                user_weight = mnist_tutorial.train(user_net, user_data[user_number], user_target[user_number],
                                                   fed_sgd=fed_sgd)
                weights.append(user_weight)
            weights_average = {}
            first_user = True
            for user_weight in weights:
                for (key, value) in user_weight.items():
                    if first_user:
                        weights_average[key] = value / active_user_count
                    else:
                        weights_average[key] += value / active_user_count
                first_user = False
            cn.load_state_dict(weights_average)
        print(f'Global epoch {c_epoch}: ', end='')
        test_losses.append(mnist_tutorial.test(cn))
    do_drawings(c_epochs, test_losses, user_count, active_user_count, user_fraction, c_rate, fed_sgd)


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

batch_size_test = 1000


def train_loader(user_count):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=60000 // user_count, shuffle=True)


def do_drawings(c_epochs, test_losses, user_count, active_user_count, user_fraction, c_rate, fed_sgd):
    l_epoch_factor = 1
    if not fed_sgd:
        l_epoch_factor = mnist_tutorial.local_epochs
    fig = plt.figure()
    train_counter = [i*(60000 // user_count) for i in range((c_epochs * active_user_count) * l_epoch_factor)]
    plt.plot(train_counter, mnist_tutorial.train_losses, color='blue')
    test_counter = [i*(60000 // user_count * active_user_count * l_epoch_factor) for i in range(c_epochs + 1)]
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.savefig(f'figure-{user_count}-{user_fraction}-{c_epochs}-{c_rate}.png')


if __name__ == '__main__':
    experiment(10, 0.3, 52, 0.5, True)
