import time

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader


class Agent:
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args

        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def local_train(self, local_model, criterion, round=None):
        """ Do a local training over the received global model """

        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.args.client_lr,
            weight_decay=self.args.wd,
            momentum=self.args.momentum
        )

        for local_epoch in range(self.args.local_ep):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):

                optimizer.zero_grad()
                inputs, labels = inputs.to(
                    device=self.args.device, non_blocking=True
                ), labels.to(device=self.args.device, non_blocking=True)
                outputs = local_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                optimizer.step()

            end = time.time()
            train_time = end - start
            print(
                "global round: %d/%d \t local epoch: %d \t client ID: %d \t loss: %.8f \t time: %.2f"
                % (
                    round,
                    self.args.rounds,
                    local_epoch + 1,
                    self.id,
                    minibatch_loss,
                    train_time,
                )
            )

        with torch.no_grad():
            return local_model
