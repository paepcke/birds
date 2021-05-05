"""
Trains a single instance of the model and saves the training results to a .jsonl file.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.multiprocessing
from datetime import datetime
from birdsong.nets import NetUtils
from evaluations import Evaluations as eval
import json
import sys


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
FILEPATH = "/home/data/birds/NEW_BIRDSONG/"
EPOCHS = 60
SEED = 42
BATCH_SIZE = 32
KERNEL_SIZE = 7
GPU = 0
# TODO: Below are params that we want to be able to change from command line
NET_NAME = 'Resnet18'
PRETRAINED = True
FREEZE_LAYERS = 0
INPUT_DIMS = (64, 64) # to match resnet18 input layer


class Training:
    """Creates and instance of the model and can train that model with specified parameter and log the results.
    :param file_path: the parent directory of the labelled samples.
    :type file_path: str
    :param epochs: the max number of epochs to train each model for.
    :type epochs: int
    :param batch_size: a value used to define the batch size of each layer.
    :type batch_size: int
    :param kernel_size: a value the defines the kernel size of the convolutional layers.
    :type kernel_size: int
    :param seed: the integer seed for training the model.
    :type seed: int
    :param net_name: the name of the type of model to use
    :type net_name: str
    :param gpu: which GPU to use. The CPU is used if gpu=None.
    :type gpu: int
    """
    def __init__(self, file_path, epochs, batch_size, kernel_size, seed=42, net_name='BasicNet', gpu=0):
        """Constructor method
        """

        # Set cwd
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        # handle seed
        if seed is not None:
            self.set_seed(seed)

        # define device used
        self.device = torch.device("cuda:" + str(gpu) if (torch.cuda.is_available() and gpu is not None) else "cpu")
        print(self.device)

        # variables
        self.EPOCHS = epochs
        self.epoch = 0
        self.BATCH_SIZE = batch_size
        # self.KERNEL_SIZE = kernel_size
        self.filepath = file_path

        # Variables set in-file: TODO: change this so that we can change this from command line
        self.net_name = NET_NAME
        self.pre_trained = PRETRAINED
        self.freeze = FREEZE_LAYERS

        # Initialize Tensorboard Writer
        self.writer = SummaryWriter()

        # configure net
        self.train_data_loader, self.val_data_loader, self.num_classes = self.import_data()
        # self.model = self.get_net(net_name, self.BATCH_SIZE, self.KERNEL_SIZE, GPU)
        # Below from Andreas' branch
        self.model = NetUtils.get_net(NET_NAME,
                                      num_classes=self.num_classes,
                                      pretrained=self.pre_trained,
                                      freeze=self.freeze,
                                      to_grayscale=True
                                      )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_func = nn.CrossEntropyLoss()

    def get_net(self, net_name, batch_size, kernel_size, gpu):
        """
        Creates and returns an instance of a specified model
        :param net_name: the name of the type of model to use
        :type net_name: str
        :param batch_size: An int used to define the batch size of each layer.
        :type batch_size: int
        :param kernel_size: An int the defines the kernel size of the convolutional layers.
        :type kernel_size: int
        :param gpu: Which GPU to use. The CPU is used if gpu=None.
        :type gpu: int
        """
        if net_name == 'BasicNet':
            return BasicNet(batch_size, kernel_size, gpu)
        if NET_NAME == 'Resnet18':  # change this to use Andreas's Resnet
            return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        # default to basic net
        else:
            return BasicNet(batch_size, kernel_size, gpu)

    def set_seed(self, seed):
        """
        set the seed for the model to allow for comparison of different models.
        :param seed: the integer seed for training the model.
        :type seed: int
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # access data
    def import_data(self):
        """ Imports the spectrograms using DataLoaders and transforms each image into a tensor.
        :return: a DataLoader containing the training data
        """

        # define the transform to be done on all images
        transform_img = transforms.Compose([
            transforms.Resize(INPUT_DIMS),  # should actually be 1:3 but broke the system
            transforms.ToTensor(),
            transforms.Grayscale()])

        # create the training DataLoader
        print(os.listdir(self.filepath+"train/"))
        # train_data = ImageFolderWithPaths(root=self.filepath + "train/", transform=transform_img)
        train_data = datasets.folder.ImageFolder(root=os.path.join(self.filepath, "train/"), transform=transform_img)
        train_data_loader = data.DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)
        # create the testing DataLoader
        # val_data = ImageFolderWithPaths(root=self.filepath + "validation/", transform=transform_img)
        val_data = datasets.folder.ImageFolder(root=os.path.join(self.filepath, "validation/"), transform=transform_img)
        val_data_loader = data.DataLoader(val_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True)

        # print useful information about the data
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(val_data))
        print("Detected Classes are: ", train_data.class_to_idx)

        num_classes = len(train_data.class_to_idx)
        return train_data_loader, val_data_loader, num_classes

    def train(self):
        """
        Trains the model until the accuracy plateaus or the maximum
        number of epochs to train has been reached. Logs the results of the training after each epoch.
        """
        # Training
        self.epoch = 0
        loss = 0
        # while the accuracy is decreasing or the number of epochs is <15, and while the number of epochs <= 100
        while (diff_avg >= 0.05 or self.epoch <= 15) and self.epoch <= 100:
            self.epoch += 1
            loss_out = 0
            # Runs batches in the training DataLoader through the model
            print("Starting epoch ", self.epoch)
            self.model.train() # prepare for training
            for batch_images, batch_labels in self.train_data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = self.loss_func(outputs, batch_labels)
                # loss_out += loss
                loss.backward()
                self.optimizer.step()
            # Testing after each epoch
            print(f"Finished epoch {self.epoch} -- Testing accuracies..." )
            # Compute predictions over training set
            train_labels, train_preds = self.predict(self.train_data_loader)
            # Compute predictions over validation set
            val_labels, val_preds = self.predict(self.val_data_loader)
            # tests if the training results match the labels in the DataLoader
            # Training Accuracy
            training_accuracy = eval.accuracy(train_labels, train_preds)
            print(f"Epoch {self.epoch}: train_acc = {training_accuracy}")
            self.writer.add_scalar("accuracy/train", training_accuracy, self.epoch)
            # Validation Accuracy
            validation_accuracy = eval.accuracy(val_labels, val_preds)
            print(f"Epoch {self.epoch}: val_acc = {validation_accuracy}")
            self.writer.add_scalar("accuracy/validation", validation_accuracy, self.epoch)

            # Compute confusion matrix
            class_names = dataloader.class_to_idx.keys()
            confusion_matrix, cm_figure, precision, recall = eval.compute_cm(val_labels, val_preds, class_names)
            self.writer.add_figure("Confusion Matrix (validation)", cm_figure, epoch=self.epoch)

        # Finished with training, so stop writing to tensorboard
        self.writer.flush()
        self.writer.close()

    def predict(self, data_loader):
        """
        Computes predictions on data over all batches. Returns test_labels and test_predictions
        :param data_loader: the DataLoader to examine
        :type data_loader: DataLoader
        :return: Python list of labels and predictions
        """
        self.model.eval()
        test_labels = []
        test_predictions = []
        # finds the number of correctly labelled
        with torch.no_grad():
            for batch_images, batch_labels in data_loader:
                outputs = self.model(batch_images)
                _, predictions = torch.max(outputs.data, 1)
                # Add batch labels and predictions to total list
                test_labels.extend(list(np.numpy(batch_labels)))
                test_predictions.extend(list(np.numpy(predictions)))
        return test_labels, test_predictions

    # return percent accuracy
    def test(self, data_loader):
        """
        calculates the percent accuracy of information in a DataLaoder.
        :param data_loader: the DataLoader to examine
        :type data_loader: DataLoader
        :return: the percent accuracy of data_loader
        """

        correct = 0
        total = 0
        with torch.no_grad():
            # two cases depending on whether or not the dataloader has path information
            if len(list(enumerate(data_loader))) == 3:
                # finds the number of correctly labelled samples
                for test_step, data in enumerate(data_loader):
                    test_images, labels = data
                    outputs = self.model(test_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            else:
                # finds the number of correctly labelled samples
                for batch_images, batch_labels in data_loader:
                    # test_images, labels, path = data
                    outputs = self.model(batch_images)
                    _, predictions = torch.max(outputs.data, 1)
                    total += len(batch_labels)
                    correct += sum(predictions == batch_labels).item()  # convert from tensor to int
        # calculate and return percent accuracy
        return 100 * correct / total

if __name__ == '__main__':
    """
    Trains a single instance of the model with the parameters specified by the constants in the file header.
    """
    if len(sys.argv) > 1:
        birdsong = Training(sys.argv[1], EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)
    else:
        birdsong = Training(FILEPATH, EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)

    birdsong.train()
