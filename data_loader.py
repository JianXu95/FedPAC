import numpy as np
from numpy.core.fromnumeric import trace
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import pdb
import os
import glob
from shutil import copyfile
import json


## --------------------------------------------------
## dataset split
## --------------------------------------------------
# MNIST
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    np.random.seed(2021)
    for i in range(num_users):
        select_set = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        dict_users[i] = list(select_set)
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    shard_per_user = 3
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def mnist_noniid_s(dataset, num_users, noniid_s=20, local_size=600):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size
    num_classes = len(np.unique(dataset.targets))
    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users

## Extended-MNIST
def emnist_noniid_s(dataset, num_users, train=True, noniid_s=20, local_size=1000):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 62 classes 62*4 =248 ~ 250
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 600
    noniid_labels_list = [[i for i in range(10)],[i for i in range(10, 36)],[i for i in range(36,62)]]
    num_classes = len(np.unique(dataset.targets))
    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    start_idxs = [0 for i in range(num_classes)]
    total_idxs = [0 for i in range(num_classes)]
    for i in range(num_classes):
        start_idxs[i] = np.where(idxs_labels[1,:]==i)[0][0]
        total_idxs[i] = len(np.where(idxs_labels[1,:]==i)[0])
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label
    noniid_num_imgs = num_per_user - num_imgs_iid

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            start = start_idxs[y]+label_used[y]
            iid_num = iid_per_label
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>total_idxs[y]:
                start = start_idxs[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%3]
        noniid_labels = noniid_labels_list[i%3]
        noniid_labels_num = len(noniid_labels)
        noniid_per_num = int(noniid_num_imgs/noniid_labels_num)
        noniid_per_num_last = noniid_num_imgs - noniid_per_num*(noniid_labels_num-1)
        # rand_label = np.random.choice(label_list, noniid_labels, replace=False)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            start = start_idxs[y]+label_used[y]
            noniid_num = noniid_per_num
            if label_cnt == noniid_labels_num:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>total_idxs[y]:
                start = start_idxs[y]
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_per_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users

# CIFAR
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(2022)
    num_classes = len(np.unique(dataset.targets))
    shard_per_user = num_classes
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    shard_per_user = 3
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users


def cifar_noniid_s(dataset, num_users, noniid_s=20, local_size=600, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size if train else 300
    num_classes = len(np.unique(dataset.targets))

    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]

    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [2000 for i in range(num_classes)] if train else [500 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users


# CINIC-10
def cinic_separate_iid(dataset1, dataset2, num_users):
    """
    Sample I.I.D. client data from CINIC-10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(2022)
    num_users1 = int(num_users//2)
    num_users2 = num_users - num_users1
    num_classes = len(np.unique(dataset1.targets))
    shard_per_user = num_classes
    imgs_per_shard1 = int(len(dataset1) / (num_users1 * shard_per_user))
    imgs_per_shard2 = int(len(dataset2) / (num_users2 * shard_per_user))
    imgs_per_shard = min(imgs_per_shard1, imgs_per_shard2)
    dict_users1 = {i: np.array([], dtype='int64') for i in range(num_users1)}
    dict_users2 = {i: np.array([], dtype='int64') for i in range(num_users2)}
    
    # divide the first dataset
    idxs_dict = {}
    for i in range(len(dataset1)):
        label = dataset1.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users1):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)
    # divide and assign
    for i in range(num_users1):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users1[i] = np.concatenate(rand_set)
    
    # divide the second dataset
    idxs_dict = {}
    for i in range(len(dataset2)):
        label = dataset2.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users2):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)
    # divide and assign
    for i in range(num_users2):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users2[i] = np.concatenate(rand_set)

    return dict_users1, dict_users2


def cinic_separate_noniid(dataset1, dataset2, num_users):
    """
    Sample I.I.D. client data from CINIC-10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(2022)
    num_users1 = int(num_users//2)
    num_users2 = num_users - num_users1
    num_classes = len(np.unique(dataset1.targets))

    shard_per_user = 4

    imgs_per_shard1 = int(len(dataset1) / (num_users1 * shard_per_user))
    imgs_per_shard2 = int(len(dataset2) / (num_users2 * shard_per_user))
    imgs_per_shard = min(imgs_per_shard1, imgs_per_shard2)
    imgs_per_shard = 100
    dict_users1 = {i: np.array([], dtype='int64') for i in range(num_users1)}
    dict_users2 = {i: np.array([], dtype='int64') for i in range(num_users2)}
    
    # divide the first dataset
    idxs_dict = {}
    for i in range(len(dataset1)):
        label = dataset1.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users1):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)
    # divide and assign
    for i in range(num_users1):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users1[i] = np.concatenate(rand_set)
    
    # divide the second dataset
    idxs_dict = {}
    for i in range(len(dataset2)):
        label = dataset2.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_users2):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)
    # divide and assign
    for i in range(num_users2):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users2[i] = np.concatenate(rand_set)

    return dict_users1, dict_users2


def cinic_separate_noniid_s(dataset1, dataset2, num_users, noniid_s=20, local_size=600, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    s1 = noniid_s/100
    s2 = noniid_s/100
    num_per_user = local_size if train else 600

    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]

    num_users1 = int(num_users//2)
    num_users2 = num_users - num_users1
    num_classes = len(np.unique(dataset1.targets))
    
    # --------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user * s1)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users1 = {i: np.array([]) for i in range(num_users1)}
    num_samples = len(dataset1)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset1.targets)
    idxs1 = np.arange(len(dataset1.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    noniid_num_imgs = num_per_user - iid_per_label*num_classes

    np.random.seed(2022)
    for i in range(num_users1):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            start = y*num_per_label_total+label_used[y]
            if (label_used[y]+iid_per_label)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users1[i] = np.concatenate((dict_users1[i], idxs[start:start+iid_per_label]), axis=0)
            label_used[y] = label_used[y] + iid_per_label
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_label = int(noniid_num_imgs/noniid_labels)
        noniid_per_num = noniid_per_label
        noniid_per_num_last = noniid_num_imgs - noniid_per_label*(noniid_labels-1)
        
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            start = y*num_per_label_total+label_used[y]
            noniid_num = noniid_per_label
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users1[i] = np.concatenate((dict_users1[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users1[i] = dict_users1[i].astype(int)
    
    # -----------------------------------------------------------------------------------
    # divide the second dataset
    num_imgs_iid = int(num_per_user * s2)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users2 = {i: np.array([]) for i in range(num_users2)}
    num_samples = len(dataset2)
    num_per_label_total = int(num_samples/num_classes)
    labels2 = np.array(dataset2.targets)
    idxs2 = np.arange(len(dataset2.targets))
    # iid labels
    idxs_labels = np.vstack((idxs2, labels2))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    noniid_num_imgs = num_per_user - iid_per_label*num_classes
    noniid_per_label = int(noniid_num_imgs/noniid_labels)

    np.random.seed(2022)
    for i in range(num_users2):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            start = y*num_per_label_total+label_used[y]
            if (label_used[y]+iid_per_label)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users2[i] = np.concatenate((dict_users2[i], idxs[start:start+iid_per_label]), axis=0)
            label_used[y] = label_used[y] + iid_per_label
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_label = int(noniid_num_imgs/noniid_labels)
        noniid_per_num = noniid_per_label
        noniid_per_num_last = noniid_num_imgs - noniid_per_label*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users2[i] = np.concatenate((dict_users2[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users2[i] = dict_users2[i].astype(int)
    
    return dict_users1, dict_users2


## --------------------------------------------------
## loading dataset
## --------------------------------------------------
def mnist():
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    testset = datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    print("MNIST Data Loading...")
    return trainset, testset


def fmnist():
    trainset = datasets.FashionMNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    testset = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    print("Fashion-MNIST Data Loading...")
    return trainset, testset


def emnist():
    trainset = datasets.EMNIST('data', 'byclass', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    testset = datasets.EMNIST('data', 'byclass', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    print("EMNIST Data Loading...")
    return trainset, testset


def cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)  #
    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    print("CIFAR10 Data Loading...")
    return trainset, testset


def cinic10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    trainset = datasets.ImageFolder(
        root='data/CINIC-10/train', transform=transform_train)  #
    testset = datasets.ImageFolder(
        root='data/CINIC-10/test', transform=transform_test)
    print("CINIC10 Data Loading...")
    return trainset, testset


def cinic10_cifar():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    trainset = datasets.ImageFolder(
        root='data/CINIC-10-CIFAR/train', transform=transform_train)  #
    testset = datasets.ImageFolder(

        root='data/CINIC-10-CIFAR/test', transform=transform_test)
    print("CINIC10-CIFAR Data Loading...")
    return trainset, testset


def cinic10_imagenet():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                             (0.24205776, 0.23828046, 0.25874835)), 
    ])

    trainset = datasets.ImageFolder(
        root='data/CINIC-10-ImageNet/train', transform=transform_train)  #
    testset = datasets.ImageFolder(

        root='data/CINIC-10-ImageNet/test', transform=transform_test)
    print("CINIC10-ImageNet Data Loading...")
    return trainset, testset


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, index=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in index] if index is not None else [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset = []
    test_dataset = []
    user_groups_train = {}
    user_groups_test = {}
    train_loader = []
    test_loader = []
    global_test_loader = []

    if args.dataset in ['cifar', 'cifar10']:
        train_dataset, test_dataset = cifar10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data 
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data 
            user_groups_train = cifar_noniid_s(train_dataset, args.num_users, args.noniid_s, args.local_size, train=True)
            user_groups_test = cifar_noniid_s(test_dataset, args.num_users, args.noniid_s, args.local_size, train=False)
            print('non-IID Data Loading---')


    elif args.dataset == 'cinic':
        train_dataset, test_dataset = cinic10()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CINIC-10
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from CINIC-10
            user_groups_train = cifar_noniid_s(train_dataset, args.num_users, train=True)
            user_groups_test = cifar_noniid_s(test_dataset, args.num_users, train=False)

    elif args.dataset == 'cinic_img':
        train_dataset, test_dataset = cinic10_imagenet()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data 
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_test = cifar_iid(test_dataset, args.num_users)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data 
            user_groups_train = cifar_noniid_s(train_dataset, args.num_users, args.noniid_s, args.local_size)
            user_groups_test = cifar_noniid_s(test_dataset, args.num_users, args.noniid_s, args.local_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'cinic_sep':
        train_dataset1, test_dataset1 = cinic10_cifar()
        train_dataset2, test_dataset2 = cinic10_imagenet()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CINIC-10
            user_groups_train1, user_groups_train2 = cinic_separate_iid(train_dataset1, train_dataset2, args.num_users)
            user_groups_test1, user_groups_test2 = cinic_separate_iid(test_dataset1, test_dataset2, args.num_users)
        else:
            # Sample Non-IID user data from CINIC-10
            user_groups_train1, user_groups_train2 = cinic_separate_noniid_s(train_dataset1, train_dataset2, args.num_users, args.noniid_s, args.local_size, train=True)
            user_groups_test1, user_groups_test2 = cinic_separate_noniid_s(test_dataset1, test_dataset2, args.num_users, args.noniid_s, args.local_size, train=False)

    elif args.dataset in ['mnist', 'fmnist']:
        if args.dataset == 'mnist':
            train_dataset, test_dataset = mnist()
        if args.dataset == 'fmnist':
            train_dataset, test_dataset = fmnist()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data 
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data 
            user_groups_train = mnist_noniid_s(train_dataset, args.num_users, args.noniid_s, args.local_size)
            user_groups_test = mnist_noniid_s(test_dataset, args.num_users, args.noniid_s, args.local_size)
            print('non-IID Data Loading---')

    elif args.dataset == 'emnist':
        train_dataset, test_dataset = emnist()
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_test = mnist_iid(test_dataset, args.num_users)
            print('IID Data Loading---')
        else:
            # Sample Non-IID user data 
            user_groups_train = emnist_noniid_s(train_dataset, args.num_users, train=True, noniid_s=args.noniid_s)
            user_groups_test = emnist_noniid_s(test_dataset, args.num_users, train=False, noniid_s=args.noniid_s)
            print('non-IID Data Loading---')


    ## --------------------------------------------------------------------------------------------------------
    ## data allocation
    if args.dataset not in ['cinic_sep']:
        for idx in range(args.num_users):
            loader1 = DataLoader(DatasetSplit(train_dataset, user_groups_train[idx]),
                                batch_size=args.local_bs, shuffle=True)
            loader2 = DataLoader(DatasetSplit(test_dataset, user_groups_test[idx]),
                                batch_size=args.local_bs, shuffle=False)
            train_loader.append(loader1)
            test_loader.append(loader2)

        global_test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

    elif args.dataset == 'cinic_sep':
        num_users1 = int(args.num_users//2)
        num_users2 = args.num_users - num_users1
        for idx in range(num_users1):
            loader1 = DataLoader(DatasetSplit(train_dataset1, user_groups_train1[idx]),
                                batch_size=args.local_bs, shuffle=True)
            loader2 = DataLoader(DatasetSplit(test_dataset1, user_groups_test1[idx]),
                                batch_size=args.local_bs, shuffle=False)
            train_loader.append(loader1)
            test_loader.append(loader2)
        for idx in range(num_users2):
            loader1 = DataLoader(DatasetSplit(train_dataset2, user_groups_train2[idx]),
                                batch_size=args.local_bs, shuffle=True)
            loader2 = DataLoader(DatasetSplit(test_dataset2, user_groups_test2[idx]),
                                batch_size=args.local_bs, shuffle=False)
            train_loader.append(loader1)
            test_loader.append(loader2)

        global_test_loader = DataLoader(test_dataset1, batch_size=args.local_bs, shuffle=False)

    else:
        raise NotImplementedError()

    return train_loader, test_loader, global_test_loader


if __name__ =='__main__':
    pass
