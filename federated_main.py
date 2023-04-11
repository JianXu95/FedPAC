import numpy as np
import torch
import torch.nn as nn
import math
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST
from options import args_parser
import tools
import copy
import time

if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(device)
    # load dataset and user groups
    train_loader, test_loader, global_test_loader = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
        global_model = CifarCNN(num_classes=args.num_classes).to(device)
        args.lr = 0.02
    elif args.dataset == 'fmnist':
        global_model = CNN_FMNIST().to(device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError()
    

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, train_acc = [], []
    test_acc = []
    local_accs1, local_accs2 = [], []
#======================================================================================================#
    local_clients = []
    for idx in range(args.num_users):
        local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx], 
                                        model=copy.deepcopy(global_model)))

    for round in range(args.epochs):
        loss1, loss2, local_acc1, local_acc2 = train_round_parallel(args, global_model, local_clients, round)
        train_loss.append(loss1)
        print("Train Loss: {}, {}".format(loss1, loss2))
        print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)


