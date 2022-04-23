# https://github.com/AshwinRJ/Federated-Learning-PyTorch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import numpy as np
import random

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


# class to train a validate the client model
class LocalUpdate(object):
    def __init__(self, dataset, idxs, gpu=1, optimizer="sgd", local_batch_size=10,
                 lr=0.01, local_epochs=10, loss_function="NLLLoss"):

        self.gpu = gpu
        self.device = 'cuda' if self.gpu else 'cpu'
        self.optimizer = optimizer
        self.local_batch_size = local_batch_size
        self.lr = lr
        self.local_epochs = local_epochs

        # Default criterion set to NLL loss function
        if loss_function == "NLLLoss":
            self.criterion = nn.NLLLoss()
        if loss_function == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()

        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs))  # get train, valid sets

    # for REPRODUCIBILITY https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def train_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.local_batch_size, shuffle=True, generator=g,
                                 worker_init_fn=self.seed_worker)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(int(len(idxs_test) / 10), 1), shuffle=False, generator=g,
                                worker_init_fn=self.seed_worker)

        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                        momentum=0.9, weight_decay=5e-4)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=1e-4)

        # loss_list = []
        # [epoch, train_loss, valid_loss, train_acc, valid_acc]
        for epoch in range(1, self.local_epochs + 1):
            # keep track of training and validation loss
            train_loss = 0.0
            correct_train = 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # move tensors to GPU if CUDA is available
                images = images.to(self.device)
                labels = labels.to(self.device)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(images)
                # calculate the batch loss
                loss = self.criterion(output, labels)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()

                # update training loss
                train_loss += (loss.data.item() * images.shape[0])
                # print('outputs on which to apply torch.max ', prediction)
                # find the maximum along the rows, use dim=1 to torch.max()
                _, predicted_outputs = torch.max(output.data, 1)
                # Update the running corrects
                correct_train += (predicted_outputs == labels).float().sum().item()

            # calculate average losses
            train_loss = train_loss / len(self.trainloader.dataset)
            # calculate accuracies
            train_acc = correct_train / len(self.trainloader.dataset)

            epoch_loss.append(train_loss)

            print('| Global Round : {} | Local Epoch : {} | Train Loss: {:.4f} | Train Accuracy: {:.2f}'.format(
                global_round, epoch, train_loss, train_acc))

        print('| Global Round : {} | Average Train Loss: {:.4f} '.format(
            global_round, sum(epoch_loss) / len(epoch_loss)
        ))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)  # avg_loss ???

    def inference(self, model):
        """
        Returns the inference accuracy and loss.
        """

        model.eval()
        # loss, total, correct = 0.0, 0.0, 0.0
        valid_loss = 0.0
        correct_valid = 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            # move tensors to GPU if CUDA is available
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Inference
            output = model(images)
            loss = self.criterion(output, labels)
            valid_loss += (loss.item() * images.shape[0])

            # Prediction
            _, predicted_outputs = torch.max(output.data, 1)
            correct_valid += (predicted_outputs == labels).float().sum().item()

        # calculate average losses
        valid_loss = valid_loss / len(self.testloader.dataset)
        # calculate accuracies
        valid_acc = correct_valid / len(self.testloader.dataset)

        return valid_acc, valid_loss


# def test_inference(model, test_dataset, gpu=1, local_batch_size=10, loss_function="NLLLoss"):
#     """
#     Returns the test accuracy and loss.
#     """
#
#     model.eval()
#     loss, total, correct = 0.0, 0.0, 0.0
#
#     device = 'cuda' if gpu else 'cpu'
#     if loss_function == "NLLLoss":
#         criterion = nn.NLLLoss()
#     if loss_function == "CrossEntropyLoss":
#         criterion = nn.CrossEntropyLoss()
#
#     testloader = DataLoader(test_dataset, batch_size=local_batch_size,
#                             shuffle=False, generator=g)
#
#     for batch_idx, (images, labels) in enumerate(testloader):
#         images, labels = images.to(device), labels.to(device)
#
#         # Inference
#         output = model(images)
#         batch_loss = criterion(output, labels)
#         loss += batch_loss.item()
#
#         # Prediction
#         _, pred_labels = torch.max(output, 1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)
#
#     # accuracy = correct / len(testloader.dataset)
#     accuracy = correct / total
#     return accuracy, loss


def test_inference(model, test_dataset, gpu=1, local_batch_size=10, loss_function="NLLLoss"):
    """
    Returns the test accuracy and loss.
    """

    model.eval()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    device = 'cuda' if gpu else 'cpu'
    if loss_function == "NLLLoss":
        criterion = nn.NLLLoss()
    if loss_function == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()

    testloader = DataLoader(test_dataset, batch_size=local_batch_size,
                            shuffle=False, generator=g)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # Inference
        output = model(images)
        loss = criterion(output, labels)
        test_loss += (loss.data.item() * images.shape[0])

        # Prediction
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not gpu else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(len(images)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(testloader.dataset)

    accuracy = np.sum(class_correct) / np.sum(class_total)

    return accuracy, test_loss
