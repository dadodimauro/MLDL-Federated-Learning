# https://github.com/AshwinRJ/Federated-Learning-PyTorch
import collections

import torch
from numpy import random
from torchvision import datasets
import torchvision.transforms as transforms

import copy

import numpy as np
import random

torch.manual_seed(0)

g = torch.Generator()
g.manual_seed(0)

np.random.seed(0)


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


def get_server(train_dataset):
    server_data = []
    server_id = []
    server_labels = []
    for i in range(len(train_dataset)):
        server_data.append(train_dataset[i][0])
        server_labels.append(train_dataset[i][1])
        server_id.append(i)
    return server_data, server_labels, server_id


# Initialize the dict_labels
def get_dict_labels(server_id, server_labels):
    dict_labels = {}
    num_classes = 10
    labels = np.arange(0, num_classes)  # the 10 classes we have : from 0 to 9
    for label in labels:
        if label not in dict_labels:
            dict_labels[label] = []
    # We create a dictionary of labels in which we have as keys the labels and the values the indexes of the images
    for i in range(len(server_labels)):
        for label in labels:
            if label == server_labels[i]:
                dict_labels[label].append(server_id[i])
    return dict_labels


# Define for n clients how many images to take
def random_number_images(n, server_id):
    # for REPRODUCIBILITY https://pytorch.org/docs/stable/notes/randomness.html
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    left = len(server_id)
    items = []
    while n > 1:
        average = left / n
        item = int(random.uniform(10, average * 3.1))
        left = left - item
        items.append(item)
        n = n - 1
    return np.array(items)


def cifar_iid_balanced(server_id, server_labels, num_users):
    # num_users = 100
    users = np.arange(0, num_users)
    num_items_balanced = int(len(server_id) / num_users)
    dict_users = collections.defaultdict(dict)
    labels = np.arange(0, 10)
    dict_labels = get_dict_labels(server_id, server_labels)
    all_idxs = [i for i in range(len(server_id))]
    new_dict = {}
    nets_cls_counts = collections.defaultdict(dict)
    for user in users:
        for label in labels:
            dict_users[user][label] = set(np.random.choice(dict_labels[label],
                                                           int(num_items_balanced / 10), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[user][label])
            nets_cls_counts[user][label] = len(list(dict_users[user][label]))
        new_dict[user] = set().union(dict_users[user][0], dict_users[user][1], dict_users[user][2],
                                     dict_users[user][3], dict_users[user][4], dict_users[user][5], dict_users[user][6],
                                     dict_users[user][7], dict_users[user][8], dict_users[user][9])

    return server_labels, new_dict, nets_cls_counts


def cifar_iid_unbalanced(server_id, server_labels, num_users):
    # num_users = 100
    users = np.arange(0, num_users)
    num_items_unbalanced = random_number_images(num_users + 1, server_id)
    dict_users = collections.defaultdict(dict)
    labels = np.arange(0, 10)
    dict_labels = get_dict_labels(server_id, server_labels)
    all_idxs = [i for i in range(len(server_id))]
    new_dict = {}
    nets_cls_counts = collections.defaultdict(dict)
    for user in users:
        for label in labels:
            dict_users[user][label] = set(np.random.choice(dict_labels[label],
                                                           int(num_items_unbalanced[user] / 10), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[user][label])
            nets_cls_counts[user][label] = len(list(dict_users[user][label]))
        new_dict[user] = set().union(dict_users[user][0], dict_users[user][1], dict_users[user][2], dict_users[user][3],
                                     dict_users[user][4], dict_users[user][5], dict_users[user][6], dict_users[user][7],
                                     dict_users[user][8], dict_users[user][9])

    return server_labels, new_dict, nets_cls_counts


def cifar_non_iid_unbalanced(server_id, server_labels, num_users):
    # num_users = 100
    users = np.arange(0, num_users)
    num_items_unbalanced = random_number_images(num_users + 1, server_id)
    # it respresents the number of images each user has for the unbalanced split of the dataset

    dict_users = {}
    all_idxs = [i for i in range(len(server_id))]
    for user in users:
        dict_users[user] = np.array(np.random.choice(all_idxs, num_items_unbalanced[user], replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[user]))
    counting = {}
    net_cls_counts = {}
    for i, dataidx in dict_users.items():
        counting = {}
        for each in dataidx:
            if server_labels[each] in counting:
                x = int(counting[server_labels[each]])
                counting[server_labels[each]] = x + 1
            else:
                counting[server_labels[each]] = 1
        sortedDict = dict(sorted(counting.items(), key=lambda x: x[0]))
        net_cls_counts[i] = sortedDict
    return server_labels, dict_users, net_cls_counts


def cifar_non_iid_balanced(server_id, server_labels, num_users):
    # num_users = 100
    users = np.arange(0, num_users)
    num_items_balanced = int(
        len(server_id) / num_users)  # it respresents the number of images each user has for the balanced split of
                                     # the dataset; each user has the same number of images
    dict_users = {}
    labels = np.arange(0, 10)
    all_idxs = [i for i in range(len(server_id))]
    list_labels = []
    for user in users:
        dict_users[user] = np.array(np.random.choice(all_idxs, num_items_balanced, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[user]))
    counting = {}
    net_cls_counts = {}
    for i, dataidx in dict_users.items():
        counting = {}
        for each in dataidx:
            if server_labels[each] in counting:
                x = int(counting[server_labels[each]])
                counting[server_labels[each]] = x + 1
            else:
                counting[server_labels[each]] = 1
                sortedDict = dict(sorted(counting.items(), key=lambda x: x[0]))
                net_cls_counts[i] = sortedDict

    return server_labels, dict_users, net_cls_counts


def get_dataset(iid, unbalanced, num_users):
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

    server_data, server_labels, server_id = get_server(train_dataset)

    if iid:
        if unbalanced:
            user_groups = cifar_iid_unbalanced(server_id, server_labels, num_users)
        else:
            user_groups = cifar_iid_balanced(server_id, server_labels, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unbalanced:
            # Chose unequal splits for every user
            user_groups = cifar_non_iid_unbalanced(server_id, server_labels, num_users)
        else:
            # Chose equal splits for every user
            user_groups = cifar_non_iid_balanced(server_id, server_labels, num_users)

    return train_dataset, test_dataset, user_groups[1]


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
