# train.py
# Define the class for training

import time
import numpy as np
import torch
import os
import json
import pandas as pd
from sklearn.metrics import classification_report


class Trainer():
    '''
    Trainer class
    '''

    def __init__(self, model, epochs, train_dataloader, val_dataloader, test_dataloader,
                 criterion, optimizer, lr_decay, lr_scheduler, device, model_dir,
                 model_name):
        '''
        parameters:
        - model: torch.nn.module, model to be trained
        - epochs: number of epochs
        - train_dataloader: DataLoader that contains training data
        - val_dataloader:  DataLoader that contains validation data
        - test_dataloader:  DataLoader that contains test data
        - criterion: loss function, expected to be torch.nn.CrossEntropyLoss()
        - optimizer: optimizer used in training, expected to be classes in torch.optim
        - lr_decay: whether to use learning rate decay
        - lr_scheduler: learning rate scheduler, expected to be classes in torch.optim.scheduler
        - device: the training device, expected to be torch.device
        - model_dir: path to save the model
        - model_name: model name, str
        '''

        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_info = "{}_{}_{}_{}".format(model.dropout, model.hidden_dim,
                                               model.num_layers, model.pretrain)
        self.model.to(self.device)
        self.embed_size = self.model.embed_size
        self.log_path = os.path.join(self.model_dir, "train_log_{}.json".format(self.model_info))

    def train(self):
        '''
        Train the model.
        During each epoch:
        - save the training, validation message (including the loss, the learning rate and time) into a json file.
        - save the model if the validation loss is the best.
        '''

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(param.numel() for param in self.model.parameters())
        model_para = 'total parameters: {}, trainable parameters: {}'.format(total, trainable)
        print(model_para)

        model_setting = '''
        dropout rate: {}, hidden size: {}, 
        hidden layer: {}, pretrained: {}
        '''.format(self.model.dropout, self.model.hidden_dim,
                   self.model.num_layers, self.model.pretrain)
        print(model_setting)

        optimizer_param = self.optimizer.state_dict()['param_groups'][0]
        train_setting = '''
        initial learning rate: {}, weight_decay: {}
        use lr schedular: {}, total epochs: {}
        '''.format(optimizer_param['initial_lr'], optimizer_param['weight_decay'],
                   self.lr_decay, self.epochs)
        print(train_setting)

        schedular_info = self.lr_scheduler.state_dict()
        lr_schedular_setting = '''
        step size: {}, gamma: {}
        '''.format(schedular_info['step_size'], schedular_info['gamma'])
        print(lr_schedular_setting)

        begin_message = "Begin training, total epochs: {}".format(self.epochs)
        print(begin_message)

        logs = [model_para, model_setting, train_setting, lr_schedular_setting,
                begin_message]  # to save all information
        best_loss = 10

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            val_loss = []
            start_time = time.time()
            train_log = {}  # save training logs

            lr_message = 'learning rate of epoch {}: {}'.format(epoch + 1, self.lr_scheduler.get_last_lr()[0])
            train_log['lr_message'] = lr_message

            # training
            for i, batch_data in enumerate(self.train_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            epoch_train_loss = np.mean(train_loss)
            train_message = '[ Epoch {}, Train ] | Loss:{:.5f} Time:{:.6f}'.format(epoch + 1,
                                                                                   epoch_train_loss,
                                                                                   time.time() - start_time)
            print(train_message)

            # validation
            self.model.eval()
            start_time = time.time()
            with torch.no_grad():
                for i, batch_data in enumerate(self.val_dataloader, 1):
                    inputs = batch_data[0].to(self.device)
                    labels = batch_data[1].to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss.append(loss.item())

            epoch_val_loss = np.mean(val_loss)
            val_message = '[ Epoch {}, Val ] | Loss:{:.5f} Time:{:.6f}'.format(epoch + 1, epoch_val_loss,
                                                                               time.time() - start_time)
            print(val_message)
            flag = False
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                save_message = 'save model with val loss {:.3f}'.format(best_loss)
                print(save_message)
                flag = True
                torch.save(self.model, "{}/{}_{}.pt".format(self.model_dir, self.model_name, self.model_info))

            if self.lr_decay:
                self.lr_scheduler.step()

            train_log["epoch"] = epoch + 1
            train_log["train_message"] = train_message
            train_log["val_message"] = val_message
            train_log["epoch_train_loss"] = epoch_train_loss
            train_log["epoch_val_loss"] = epoch_val_loss
            if flag:
                train_log["save_message"] = save_message
            logs.append(train_log)
            with open(self.log_path, "w") as fp:
                json.dump(logs, fp)
        print("Finish training!")

    def test(self):
        model = torch.load("{}/{}_{}.pt".format(self.model_dir, self.model_name, self.model_info))
        predictions = []
        y_labels = []
        test_loss = []

        print("Begin testing")
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_dataloader, 1):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss.append(loss.item())
                pred = outputs.argmax(dim=1)
                predictions.extend(pred.cpu().numpy().tolist())
                y_labels.extend(labels.cpu().numpy().tolist())

        epoch_test_loss = np.mean(test_loss)
        test_message = "Test loss: {}".format(epoch_test_loss)
        report = classification_report(y_labels, predictions, digits=4, zero_division=1)
        print(test_message)

        # make the classification report more readable
        report = report.splitlines()
        columns = ['class'] + report[0].split()
        col_1, col_2, col_3, col_4, col_5 = [], [], [], [], []
        for row in report[1:]:
            if len(row.split()) != 0:
                row = row.split()
                if len(row) < 5:
                    col_1.append(row[0])
                    col_2.append('')
                    col_3.append('')
                    col_4.append(row[1])
                    col_5.append(row[2])
                elif len(row) > 5:
                    col_1.append(row[0] + ' ' + row[1])
                    col_2.append(row[2])
                    col_3.append(row[3])
                    col_4.append(row[4])
                    col_5.append(row[5])
                else:
                    col_1.append(row[0])
                    col_2.append(row[1])
                    col_3.append(row[2])
                    col_4.append(row[3])
                    col_5.append(row[4])
        result = pd.DataFrame()
        result[columns[0]] = col_1
        result[columns[1]] = col_2
        result[columns[2]] = col_3
        result[columns[3]] = col_4
        result[columns[4]] = col_5
        print("——————Test——————")
        print(result)

        # save results
        result.to_csv("{}/{}_{}.csv".format(self.model_dir, 'result', self.model_info), index=False)

        with open(self.log_path, "r") as fp:
            logs = json.load(fp)
        logs.append(test_message)
        with open(self.log_path, "w") as fp:
            json.dump(logs, fp)