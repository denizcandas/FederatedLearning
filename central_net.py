import torch
import random
import mnist_tutorial
import sys
import asyncio
import pickle
user_count = int(sys.argv[1])


async def experiment(user_count, user_fraction, c_epochs, c_rate):
    active_user_count = int(user_count * user_fraction)
    network_list_reader = []
    network_list_writer = []

    cn = mnist_tutorial.Net()  # central_network
    print(len(pickle.dumps((cn.state_dict(), 'a very well thought out protocol'))))
    optimizer = torch.optim.SGD(cn.parameters(), lr=c_rate, momentum=0.5)

    for user in range(user_count):
        reader, writer = await asyncio.open_connection('localhost', 8000 + user, limit=1024 * 128)
        network_list_reader.append(reader)
        network_list_writer.append(writer)

    for c_epoch in range(c_epochs):
        print(f'Global epoch {c_epoch} has started')
        chosen_users = random.sample(range(user_count), active_user_count)
        for user in chosen_users:
            binary_state_dict = pickle.dumps((cn.state_dict(), 'a very well thought out protocol'))
            network_list_writer[user].write(binary_state_dict)

        for user in chosen_users:
            binary_gradients = await network_list_reader[user].readuntil(
                separator=b'a very well thought out protocol\x94\x86\x94.')
            gradients = pickle.loads(binary_gradients)[0]
            for cn_param, user_gradient in zip(cn.named_parameters(), gradients):
                if cn_param[1].grad is not None:
                    cn_param[1].grad += user_gradient.clone() / active_user_count
                else:
                    cn_param[1].grad = user_gradient.clone() / active_user_count
        optimizer.step()
        optimizer.zero_grad()

        print(f'Global epoch {c_epoch}: ', end='')
        mnist_tutorial.test(cn)


asyncio.run(experiment(user_count, 0.4, 10, 0.5))
