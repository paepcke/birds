"""
A file that uses sci-kit learn's StratifiedKFold module to split the training data
and then uses cross validation to train and validate models with different sets of data.
"""
from sklearn.model_selection import StratifiedKFold
from training import Training
import numpy as np
import os
import shutil
import sys

# the location of the samples
FILEPATH = "/home/data/birds/NEW_BIRDSONG/ALL_SPECTROGRAMS/"
EPOCHS = 100
SEED = 42
BATCH_SIZE = 16
KERNEL_SIZE = 5
NET_NAME = 'BasicNet'
GPU = 0


class cross_fold_split_and_validation:
    """
    A class that creates KFoldStratified splits of the data and then trains separate models and evaluates them with
    cross-validation.

    :param filepath: the parent directory of the labelled samples.
    :type filepath: str
    :param epochs: the max number of epochs to train each model for.
    :type epochs: int
    :param seed: the integer seed for training the model.
    :type seed: int
    :param batch_size: An int used to define the batch size of each layer.
    :type batch-size: int
    :param kernel_size: An int the defines the kernel size of the convolutional layers.
    :type kernel_size: int
    :param net: the name of the type of model to use
    :type net: str
    :param processor: The number of the GPU to use. The CPU is used if None.
    :type processor: int
    """
    def __init__(self, filepath, epochs, seed, batch_size=32, kernel_size=5, net='BasicNet', processor=None):
        """Constructor method
        """
        #initialize all the fields
        # these are just passed to the model
        self.file_path = filepath
        self.max_epochs = epochs
        self.seed = seed
        self.bs = batch_size
        self.ks = kernel_size
        self.model = net
        self.gpu = processor
        # these are used for tracking all the samples and splitting them
        self.X = []  # samples
        self.y = []  # labels
        self.file_list = []  # all of the samples
        self.folder_list = []  # all of the classes
        self.ident = 0  # the number of classes
        self.count = 0  # the number of total samples
        self.train_index = 0
        self.test_index = 0

    def count_and_label(self):
        """Counts the number of classes, counts the number of samples in each class,
        stores the samples with their associated labels.
        """
        # for each class
        for folder in os.listdir(self.file_path):
            self.folder_list.append(folder)
            # for each sample in the class
            for file in os.listdir(self.file_path + folder):
                self.file_list.append(file)
                self.X.append(self.count)
                self.y.append(self.ident)
                self.count += 1
            self.ident += 1
        # X is the sample, y is the label
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def run(self):
        """
        Splits the train and test data using the StratifiedKFold function from sklearn
        and then trains and evaluetes the model for each split.
        """
        # Perform stratified K Fold split
        skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        # enumerate splits and train models
        num = 0
        for train_index, test_index in skf.split(self.X, self.y):
            num += 1
            print("running split:", num)
            # gets the train and test data for this split
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.ident = 0
            self.count = 0
            # create folders with just the train and test data for this split
            for folder in os.listdir(self.file_path):
                self.delete_folder_contents(self.file_path + "../train/" + self.folder_list[self.ident])
                self.delete_folder_contents(self.file_path + "../validation/" + self.folder_list[self.ident])
                # populate each class's folder within the train and validation folders
                for file in os.listdir(self.file_path + folder):
                    if train_index < X_train.shape[0] and X_train[train_index] == self.count:
                        shutil.copyfile(self.file_path + folder + "/" + file,
                                        self.file_path + "../train/" + self.folder_list[self.ident] + "/" + file)
                        train_index += 1
                    if test_index < X_test.shape[0] and X_test[test_index] == self.count:
                        shutil.copyfile(self.file_path + folder + "/" + file,
                                        self.file_path + "../validation/" + self.folder_list[self.ident] + "/" + file)
                        test_index += 1
                    self.count += 1
                self.ident += 1
            # create and train the model with this split of data
            birdsong = Training(self.file_path + "../", self.max_epochs, self.bs, self.ks, self.seed, self.model,
                                self.gpu)
            birdsong.train()

    def delete_folder_contents(self, folder):
        """Deletes the contents of temporary folder in between runs
        """
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            # checks if it is a real file/folder and removes a single file or a whole file tree
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    """The main method for the manager module.
    """
    if len(sys.argv) > 1:
        cfsv = cross_fold_split_and_validation(sys.argv[1], EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE, NET_NAME, GPU)
    else:
        cfsv = cross_fold_split_and_validation(FILEPATH, EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE, NET_NAME, GPU)

    cfsv.count_and_label()
    cfsv.run()
