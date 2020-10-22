# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import StratifiedKFold
from training import Training
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import os
import torchvision
from pathlib import Path
import shutil

# FILEPATH = "/Users/amyd/Desktop/Projects/birds/First_Test/"
FILEPATH = "/home/data/birds/NEW_BIRDSONG/ALL_SPECTROGRAMS/"
# FILEPATH = "/Users/LeoGl/PycharmProjects/bird/First_Test/test/"
EPOCHS = 1
SEED = 42
BATCH_SIZE = 16
KERNEL_SIZE = 5
NET_NAME = 'BasicNet'
# NET_NAME = 'Resnet18'
GPU = None


def delete_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    # prepare data sets
    X = []
    y = []
    file_list = []
    folder_list = []
    ident = 0
    count = 0

    for folder in os.listdir(FILEPATH):
        folder_list.append(folder)
        for file in os.listdir(FILEPATH + folder):
            file_list.append(file)
            X.append(count)
            y.append(ident)
            count += 1
        ident += 1
    X = np.array(X)
    y = np.array(y)
    # prepare cross validation
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    # enumerate splits
    num = 0
    for train_index, test_index in skf.split(X, y):
        num +=1
        print("running split:", num)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_index = 0
        test_index = 0
        ident = 0
        count = 0
        for folder in os.listdir(FILEPATH):
            delete_folder_contents(FILEPATH + "../train/" + folder_list[ident])
            delete_folder_contents(FILEPATH + "../validation/" + folder_list[ident])
            for file in os.listdir(FILEPATH + folder):
                if train_index < X_train.shape[0] and X_train[train_index] == count:
                    shutil.copyfile(FILEPATH + folder + "/" + file,
                              FILEPATH + "../train/" + folder_list[ident] + "/" + file)
                    train_index += 1
                if test_index < X_test.shape[0] and X_test[test_index] == count:
                    shutil.copyfile(FILEPATH + folder + "/" + file,
                              FILEPATH + "../validation/" + folder_list[ident] + "/" + file)
                    test_index += 1
                count += 1
            ident += 1
        birdsong = Training(FILEPATH + "../", EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)
        birdsong.train()
