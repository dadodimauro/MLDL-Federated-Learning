import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import GTKutils

# TODO the train accuracy is wrong

class GKTServerTrainer:
    def __init__(self, model, num_users, lr, server_local_epochs, device,
                 optimizer="sgd", temperature=3.0):
        self.model = model
        self.model.train()
        self.model_params = self.model.parameters()

        self.num_users = num_users
        self.lr = lr
        self.epochs = server_local_epochs
        self.device = device

        self.temperature = temperature

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model_params, lr=self.lr, momentum=0.9,
                                             nesterov=True, weight_decay=5e-4)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model_params, lr=self.lr,
                                              weight_decay=0.0001, amsgrad=True)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = GTKutils.KL_Loss(self.temperature)

        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feature_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # key: client_index; value: logits_dict
        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feauture_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()

        self.best_acc = 0.0  # used to keep track of the best accuracy model state and also for scheduler

        self.loss_list = []
        self.acc_list = []

        self.flag_client_model_uploaded_dict = dict()  # to check if all clients sent their logits
        for idx in range(self.num_users):
            self.flag_client_model_uploaded_dict[idx] = False

        # self.distill_on_server = distill_on_server
        # self.aplha = alpha

    def get_loss_acc_list(self):
        return self.loss_list, self.acc_list

    # collect client data
    def add_local_trained_result(self, idx, extracted_feature_dict, logits_dict, labels_dict,
                                 extracted_feature_dict_test, labels_dict_test):
        print("add model - client_id = %d" % idx)

        # TODO not needed to store
        self.client_extracted_feature_dict[idx] = extracted_feature_dict
        self.client_logits_dict[idx] = logits_dict
        self.client_labels_dict[idx] = labels_dict
        self.client_extracted_feauture_dict_test[idx] = extracted_feature_dict_test
        self.client_labels_dict_test[idx] = labels_dict_test

        self.flag_client_model_uploaded_dict[idx] = True

    # TODO remove
    # check if all data from clients is received
    def check_whether_all_receive(self):
        for idx in range(self.num_users):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.num_users):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    # TODO look at train_large_model_on_the_server()
    def get_global_logits(self, client_index):
        return self.server_logits_dict[client_index]

    def train(self, communication_round, idxs_chosen_users):
        # train according to the logits from the client
        self.train_and_eval(communication_round, self.epochs, idxs_chosen_users)

        # adjust the learning rate based on the number of epochs.
        # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler.step(self.best_acc, epoch=communication_round)

        # clean all client data (in order to avoid allocating too much data in RAM)
        # TODO check if this loop in necessary
        for client_index in idxs_chosen_users:
            self.client_extracted_feature_dict[client_index].clear()
            self.client_logits_dict[client_index].clear()
            self.client_labels_dict[client_index].clear()
            self.client_extracted_feauture_dict_test[client_index].clear()
            self.client_labels_dict_test[client_index].clear()
        self.client_extracted_feature_dict.clear()
        self.client_logits_dict.clear()
        self.client_labels_dict.clear()
        self.client_extracted_feauture_dict_test.clear()
        self.client_labels_dict_test.clear()

    def train_and_eval(self, round_idx, epochs, idxs_chosen_users):
        for epoch in range(1, epochs+1):
            print("train_and_eval - round_idx = %d, epoch = %d" % (round_idx, epoch))

            train_metrics = self.train_large_model_on_the_server(idxs_chosen_users)

            if epoch == epochs:  # if it is last epoch
                print(f"Train/Loss: {train_metrics['train_loss']} - epoch: {epoch}")
                print(f"Train/Accuracy: {train_metrics['train_accuracy']} - epoch: {epoch}")

                test_metrics = self.eval_large_model_on_the_server(idxs_chosen_users)

                print(f"Test/Loss: {test_metrics['test_loss']} - epoch: {epoch}")
                print(f"Test/Accuracy: {test_metrics['test_accuracy']} - epoch: {epoch}")

                test_acc = test_metrics['test_accuracy']
                if test_acc >= self.best_acc:
                    self.best_acc = test_acc

                # update loss and accuracy list
                self.loss_list.append(test_metrics['test_loss'])
                self.acc_list.append(test_metrics['test_accuracy'])

    def train_large_model_on_the_server(self, idxs_chosen_users):
        # clear the server side logits
        for key in self.server_logits_dict.keys():
            self.server_logits_dict[key].clear()
        self.server_logits_dict.clear()

        # set server model to train mode
        self.model.train()

        loss_avg = GTKutils.RunningAverage()
        acc_avg = GTKutils.RunningAverage()

        # accTop1_avg = GTKutils.RunningAverage()
        # accTop5_avg = GTKutils.RunningAverage()

        # TODO i have to use all the clients indxs every time (more and more each communication round) or i should use only the clients of this communication round??
        # for client_index in self.client_extracted_feature_dict.keys():
        for client_index in idxs_chosen_users:
            # retrieve information for each client
            extracted_feature_dict = self.client_extracted_feature_dict[client_index]
            logits_dict = self.client_logits_dict[client_index]  # needed for knowledge distillation
            labels_dict = self.client_labels_dict[client_index]

            server_logits_dict = dict()
            self.server_logits_dict[client_index] = server_logits_dict  # TODO why??

            # print(f"extracted_feature_dict len - {len(extracted_feature_dict.keys())}")
            for batch_idx in extracted_feature_dict.keys():
                # retrieve information for every batch
                batch_feature_map = torch.from_numpy(extracted_feature_dict[batch_idx]).to(self.device)
                batch_logits = torch.from_numpy(logits_dict[batch_idx]).float().to(
                    self.device)  # needed for knowledge distillation
                batch_labels = torch.from_numpy(labels_dict[batch_idx]).long().to(self.device)

                # print("running: batch_index = %d, client_index = %d" % (batch_idx, client_index))

                output_batch = self.model(batch_feature_map)

                # if self.distill_on_server == 1:
                #     loss_kd = self.criterion_KL(output_batch, batch_logits).to(self.device)
                #     loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                #     loss = loss_kd + self.alpha * loss_true
                # else:
                #     loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                #     loss = loss_true

                # compute loss
                loss = self.criterion_CE(output_batch, batch_labels).to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # update average loss and accuracy

                # TODO check if accuracy is computed correctly
                # metrics = GTKutils.accuracy(output_batch, batch_labels, topk=(1, 5))
                # acc_avg.update(metrics[0].item())
                # accTop1_avg.update(metrics[0].item())
                # accTop5_avg.update(metrics[1].item())

                # find the maximum along the rows, use dim=1 to torch.max()
                _, predicted_outputs = torch.max(output_batch.data, 1)
                x = (predicted_outputs == batch_labels).float().sum().item() * 100 / batch_labels.size(0)
                acc_avg.update(x)

                loss_avg.update(loss.item())

                # update the logits for each client
                # Note that this must be running in the model.train() model,
                # since the client will continue the iteration based on the server logits.
                # TODO see where this server_logits_dict go
                server_logits_dict[batch_idx] = output_batch.cpu().detach().numpy()

        # compute mean of all metrics in summary
        train_metrics = {'train_loss': loss_avg.value(),
                         'train_accuracy': acc_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
        print("\n- TRAIN METRICS: " + metrics_string + "\n")

        return train_metrics

    def eval_large_model_on_the_server(self, idxs_chosen_users):
        # set model to evaluation mode
        self.model.eval()

        loss_avg = GTKutils.RunningAverage()
        acc_avg = GTKutils.RunningAverage()
        # accTop1_avg = GTKutils.RunningAverage()
        # accTop5_avg = GTKutils.RunningAverage()

        # deactivate the autograd engine in order to increase performance
        with torch.no_grad():
            # for client_index in self.client_extracted_feauture_dict_test.keys():
            for client_index in idxs_chosen_users:
                # retrieve test information for each client
                extracted_feature_dict = self.client_extracted_feauture_dict_test[client_index]
                labels_dict = self.client_labels_dict_test[client_index]
                # print(f"extracted_feature_dict_test len - {len(extracted_feature_dict.keys())}")
                for batch_idx in extracted_feature_dict.keys():
                    # retrieve information for every batch
                    batch_feature_map = torch.from_numpy(extracted_feature_dict[batch_idx]).to(self.device)
                    batch_labels = torch.from_numpy(labels_dict[batch_idx]).long().to(self.device)

                    # print("evaluating: batch_index = %d, client_index = %d" % (batch_idx, client_index))

                    output_batch = self.model(batch_feature_map)

                    # compute loss
                    loss = self.criterion_CE(output_batch, batch_labels).to(self.device)

                    # update average loss and accuracy

                    # metrics = GTKutils.accuracy(output_batch, batch_labels, topk=(1, 5))
                    # accTop1_avg.update(metrics[0].item())
                    # accTop5_avg.update(metrics[1].item())

                    # find the maximum along the rows, use dim=1 to torch.max()
                    _, predicted_outputs = torch.max(output_batch.data, 1)
                    acc_avg.update((predicted_outputs == batch_labels).float().sum().item())

                    loss_avg.update(loss.item())

        # compute mean of all metrics in summary
        test_metrics = {'test_loss': loss_avg.value(),
                        'test_accuracy': acc_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
        print("\n- TEST METRICS: " + metrics_string + "\n")

        return test_metrics
