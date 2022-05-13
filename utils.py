# https://github.com/AshwinRJ/Federated-Learning-PyTorch

import torch
from torchvision import datasets
import torchvision.transforms as transforms

import copy

import numpy as np

torch.manual_seed(2)

g = torch.Generator()
g.manual_seed(2)

np.random.seed(2)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])  # deepcopy of the weights in a local variable
    # compute the mean
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, user_groups, idxs_users):
    """
    Returns the weighted average of the weights.
    """
    n_list = []
    for idx in idxs_users:
        n_list.append(len(user_groups[idx]))

    if len(n_list) != len(w):
        print("ERROR IN WEIGHTED AVERAGE!")

    w_avg = copy.deepcopy(w[0])  # deepcopy of the weights in a local variable

    # compute the mean
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # w_avg[key] += w_avg[key] + torch.mul(w[i][key], n_list[i]/sum(n_list))
            w_avg[key] += torch.mul(w[i][key], n_list[i]/sum(n_list)).long()

    return w_avg


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # i.i.d. selection from dataset
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 250*200=50000 -> train dataset dimension
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


# TODO change this function in order to have clients with access to only some specific data
def cifar_noniid_unbalanced(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset such that
    clients have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 50,000 training imgs --> 50 imgs/shard X 1000 shards
    num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1  # at least 50 images for a client
    max_shard = 30  # at most 1500 images for a client

    # Divide the shards into random chunks for every client
    # such that the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def get_dataset(iid=1, unbalanced=0, num_users=100):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
    ])

    # choose the training and test datasets
    train_dataset = datasets.CIFAR10('data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('data', train=False,
                                    download=True, transform=transform_test)

    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unbalanced:
            # Chose unequal splits for every user
            user_groups = cifar_noniid_unbalanced(train_dataset, num_users)
        else:
            # Chose equal splits for every user
            user_groups = cifar_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups


def exp_details(model, optimizer, lr, norm, epochs, iid, frac, local_bs, local_ep, unbalanced, num_users):
    print('\nExperimental details:')
    print(f'    Model     : {model}')
    print(f'    Optimizer : {optimizer}')
    print(f'    Learning  : {lr}')
    print(f'    Normalization  : {norm}')
    print(f'    Global Rounds   : {epochs}\n')

    print('    Federated parameters:')
    if iid:
        print('    IID')
    elif unbalanced:
        print('    Non-IID - unbalanced')
    else:
        print('    Non-IID - balanced')
    print(f'    NUmber of users  : {num_users}')
    print(f'    Fraction of users  : {frac}')
    print(f'    Local Batch size   : {local_bs}')
    print(f'    Local Epochs       : {local_ep}\n')
    return
