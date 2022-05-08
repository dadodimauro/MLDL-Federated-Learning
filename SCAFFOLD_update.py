import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random
from copy import deepcopy
from math import ceil

from tqdm import trange

from fedlab.core.client import ClientTrainer
from fedlab.utils.serialization import SerializationTool

torch.manual_seed(0)

g = torch.Generator()
g.manual_seed(0)

np.random.seed(0)


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class SCAFFOLDTrainer:
    def __init__(self, id, global_model, train_dataset, test_dataset, batch_size, idxs, lr, epochs, gpu):
        # super().__init__(deepcopy(global_model), gpu)

        self.global_model = global_model
        self.epochs = epochs
        self.lr = lr
        self.id = id
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda")
        # self.device = next(iter(self.global_model.parameters())).device

        self.c_local = [
            torch.zeros_like(param, device=self.device)
            for param in self.global_model.parameters()
            if param.requires_grad
        ]

        self.trainloader, self.testloader = self.train_test(
            train_dataset, test_dataset, list(idxs))  # get train, valid sets
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_test(self, train_dataset, test_dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(train_dataset, idxs),
                                 batch_size=self.batch_size, shuffle=True, generator=g,
                                 worker_init_fn=self.seed_worker)
        testloader = DataLoader(test_dataset,
                                batch_size=self.batch_size, shuffle=False, generator=g,
                                worker_init_fn=self.seed_worker)

        return trainloader, testloader

    def train(self, global_model_parameters, c_global):

        SerializationTool.deserialize_model(self.global_model, global_model_parameters)

        self._train(self.global_model, c_global, self.epochs)
        with torch.no_grad():
            y_delta = [torch.zeros_like(param) for param in self.global_model.parameters()]
            c_new = deepcopy(y_delta)
            c_delta = deepcopy(y_delta)  # c_+ and c_delta both have the same shape as y_delta

            # calc model_delta (difference of model before and after training)
            for y_del, param_l, param_g in zip(y_delta, self.global_model.parameters(), self.global_model.parameters()):
                y_del.data += param_l.data.detach() - param_g.data.detach()

            # update client's local control
            a = (
                    ceil(len(self.trainloader.dataset) / self.batch_size)
                    * self.epochs
                    * self.lr
            )  # ceil rounds a number UP to the nearest integer
            for c_n, c_l, c_g, diff in zip(c_new, self.c_local, c_global, y_delta):
                c_n.data += c_l.data - c_g.data - diff.data / a

            self.c_local = c_new

            # calc control_delta
            for c_d, c_n, c_l in zip(c_delta, c_new, self.c_local):
                c_d.data += c_n.data - c_l.data

        return c_delta, y_delta

    def eval(self, global_model_parameters, c_global):
        model_4_eval = deepcopy(self.global_model)
        SerializationTool.deserialize_model(model_4_eval, global_model_parameters)

        # evaluate global SCAFFOLD performance
        loss_g, acc_g = evaluate(
            model_4_eval, self.testloader, self.criterion, self.device
        )

        # localization
        self._train(model_4_eval, c_global, self.id)

        # evaluate localized SCAFFOLD performance
        loss_l, acc_l = evaluate(
            model_4_eval, self.testloader, self.criterion, self.device
        )

        return loss_g, acc_g, loss_l, acc_l

    def _train(self, model, c_global, epochs):
        model.train()

        for _ in trange(epochs, desc="client [{}]".format(self.id)):
            x, y = self.get_data_batch(train=True)
            logit = model(x)
            loss = self.criterion(logit, y)
            gradients = torch.autograd.grad(loss, model.parameters())
            with torch.no_grad():
                for param, grad, c_g, c_l in zip(
                        model.parameters(), gradients, c_global, self.c_local
                ):
                    c_g, c_l = c_g.to(self.device), c_l.to(self.device)
                    param.data = param.data - self.lr * (
                            grad.data + c_g.data - c_l.data
                    )
            self.lr *= 0.95

    def get_data_batch(self, train: bool):
        if train:
            try:
                data, targets = next(self.iter_trainloader)
            except StopIteration:
                self.iter_trainloader = iter(self.trainloader)
                data, targets = next(self.iter_trainloader)
        else:
            try:
                data, targets = next(self.iter_testloader)
            except StopIteration:
                self.iter_testloader = iter(self.testloader)
                data, targets = next(self.iter_testloader)

        return data.to(self.device), targets.to(self.device)


def evaluate(model, testloader, criterion, gpu=None):
    model.eval()
    correct = 0
    loss = 0
    if gpu is not None:
        model = model.to(gpu)
    for x, y in testloader:
        if gpu is not None:
            x, y = x.to(gpu), y.to(gpu)

        logit = model(x)
        loss += criterion(logit, y)

        pred_y = torch.softmax(logit, -1).argmax(-1)
        correct += torch.eq(pred_y, y).int().sum()

    acc = 100.0 * (correct / len(testloader.dataset))
    return loss, acc
