# coding: utf-8

import tools
import math
import torch
from torch import nn
# ---------------------------------------------------------------------------- #

class LocalUpdate_LG_FedAvg(object):
    def __init__(self, idx, args, train_set, test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.agg_weight = self.aggregate_weight()
        # the local_weights in LG_FedAvg are representation layers
        self.w_local_keys = self.local_model.base_weight_keys

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w
    
    def local_test(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1) 
                correct += (predicted == labels).sum().item()
        acc = 100.0*correct/total
        return acc
    
    def update_local_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)
    
    def local_training(self, local_epoch):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = 0
        iter_loss = []
        model.zero_grad()

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        local_p=[]
        global_p=[]
        for name, p in model.named_parameters():
            if name in self.w_local_keys:
                local_p += [p]
            else:
                global_p += [p]
        optimizer = torch.optim.SGD([ {'params': local_p, 'lr':self.args.lr},
                                      {'params': global_p, 'lr':self.args.lr}
                                    ], lr=self.args.lr, momentum=0.5, weight_decay=0.0005)
        if local_epoch>0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    _, output = model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        else:
            data_loader = iter(self.train_data)
            iter_num = self.args.local_iter
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        round_loss1 = iter_loss[0]
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), round_loss1, round_loss2, acc1, acc2


