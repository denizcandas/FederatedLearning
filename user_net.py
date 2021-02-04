import torch
import torchvision
import mnist_tutorial
import sys
import asyncio
import pickle
user_number = int(sys.argv[1])


def train_loader(user_count):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])), batch_size=60000 // user_count, shuffle=False)


for batch_idx, (data, target) in enumerate(train_loader(10)):
    if batch_idx == user_number:
        print('Preparing data for user', batch_idx)
        user_data = data
        user_target = target
        print('Data loaded')
        break
else:
    raise Exception('No data could be loaded')
user_net = mnist_tutorial.Net()


async def user_net_routine(reader, writer):
    print('Connected to the central network')
    try:
        while True:
            print('Waiting for weights')
            cn_state_dict_binary = await reader.readuntil(separator=b'a very well thought out protocol\x94\x86\x94.')
            print('Received weights')
            cn_state_dict = pickle.loads(cn_state_dict_binary)[0]
            user_net.load_state_dict(cn_state_dict)
            print('Starting training')
            gradients = mnist_tutorial.train(user_net, user_data, user_target)
            gradients_binary = pickle.dumps((gradients, "a very well thought out protocol"))
            print('Transmitting results')
            writer.write(gradients_binary)
            print('Sent gradients and loss')
    except:
        print('Connection lost')


async def main():
    server = await asyncio.start_server(user_net_routine, 'localhost', 8000 + user_number, limit=1024 * 128)
    async with server:
        await server.serve_forever()


asyncio.run(main())
