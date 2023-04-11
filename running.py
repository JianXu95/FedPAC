
import torch
from torch import nn
import tools
import numpy as np
import copy
import math
import json
from tools import average_weights_weighted, get_head_agg_weight, agg_classifier_weighted_p

def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg':train_round_fedavg,
                   'LG_FedAvg':train_round_lgfedavg,
                   'FedPer':train_round_fedper,
                   'Local':train_round_standalone,
                   'FedPAC':train_round_fedpac,
    }
    return Train_Round[rule]

## training methods -------------------------------------------------------------------
# local training only
def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

# vanila FedAvg
def train_round_fedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

# parameter decoupling
def train_round_lgfedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # global_weight = average_weights(local_weights)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedper(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        agg_weight.append(local_client.agg_weight)
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2

def train_round_fedpac(args, global_model, local_clients, rnd, **kwargs):
        print(f'\n---- Global Communication Round : {rnd+1} ----')
        num_users = args.num_users
        m = max(int(args.frac * num_users), 1)
        if (rnd >= args.epochs):
            m = num_users
        idx_users = np.random.choice(range(num_users), m, replace=False)
        idx_users = sorted(idx_users)

        local_weights, local_losses1, local_losses2 = [], [], []
        local_acc1 = []
        local_acc2 = []
        agg_weight = []  # aggregation weights for f
        avg_weight = []  # aggregation weights for g
        sizes_label = []
        local_protos = []

        Vars = []
        Hs = []

        agg_g = args.agg_g # conduct classifier aggregation or not

        if rnd <= args.epochs:
            for idx in idx_users:
                local_client = local_clients[idx]
                ## statistics collection
                v, h = local_client.statistics_extraction()
                Vars.append(copy.deepcopy(v))
                Hs.append(copy.deepcopy(h))
                ## local training
                local_epoch = args.local_epoch
                sizes_label.append(local_client.sizes_label)
                w, loss1, loss2, acc1, acc2, protos = local_client.local_training(local_epoch=local_epoch, round=rnd)
                local_weights.append(copy.deepcopy(w))
                local_losses1.append(copy.deepcopy(loss1))
                local_losses2.append(copy.deepcopy(loss2))
                local_acc1.append(acc1)
                local_acc2.append(acc2)
                agg_weight.append(local_client.agg_weight)
                local_protos.append(copy.deepcopy(protos))

            # get weight for feature extractor aggregation
            agg_weight = torch.stack(agg_weight).to(args.device)

            # update global feature extractor
            global_weight_new = average_weights_weighted(local_weights, agg_weight)

            # update global prototype
            global_protos = tools.protos_aggregation(local_protos, sizes_label)

            for idx in range(num_users):
                local_client = local_clients[idx]
                local_client.update_base_model(global_weight=global_weight_new)
                local_client.update_global_protos(global_protos=global_protos)

            # get weight for local classifier aggregation
            if agg_g and rnd < args.epochs:
                avg_weights = get_head_agg_weight(m, Vars, Hs)
                idxx = 0
                for idx in idx_users:
                    local_client = local_clients[idx]
                    if avg_weights[idxx] is not None:
                        new_cls = agg_classifier_weighted_p(local_weights, avg_weights[idxx], local_client.w_local_keys, idxx)
                    else:
                        new_cls = local_weights[idxx]
                    local_client.update_local_classifier(new_weight=new_cls)
                    idxx += 1

        loss_avg1 = sum(local_losses1) / len(local_losses1)
        loss_avg2 = sum(local_losses2) / len(local_losses2)
        acc_avg1 = sum(local_acc1) / len(local_acc1)
        acc_avg2 = sum(local_acc2) / len(local_acc2)

        return loss_avg1, loss_avg2, acc_avg1, acc_avg2
