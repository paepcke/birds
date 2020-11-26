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
from nets import BasicNet
import json
import sys

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
FILEPATH = "/home/data/birds/NEW_BIRDSONG/"
EPOCHS = 60
SEED = 42
BATCH_SIZE = 32
KERNEL_SIZE = 7
NET_NAME = 'BasicNet'
GPU = 0


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    :param root: the filepath of a folder containing the images.
    :type root: str
    :param root: A transformation to perform on each image.
    :type root: method
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        """Constructor method
        """
        super(datasets.ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                   transform=transform, target_transform=target_transform,
                                                   is_valid_file=is_valid_file)

        self.imgs = self.samples

    def __getitem__(self, index):
        """override the __getitem__ method. this is the method that dataloader calls

        :param index: the index of the image
        :type index: int
        :returns: the path to the file the data originates from
        """
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    
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
        self.KERNEL_SIZE = kernel_size
        self.filepath = file_path

        # configure log file
        now = datetime.now()
        self.log_filepath = now.strftime("%d-%m-%Y") + '_' + now.strftime("%H-%M") + "_K" + str(
            self.KERNEL_SIZE) + '_B' + str(self.BATCH_SIZE) + '.jsonl'
        self.configure_log()

        # configure net
        self.model = self.get_net(net_name, self.BATCH_SIZE, self.KERNEL_SIZE, GPU)
        self.train_data_loader, self.test_data_loader = self.import_data()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
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
        """ imports the spectrograms using DataLoaders and transforms each image into a tensor.

        :return: a DataLoader containing the training data
        """

        # define the transform to be done on all images
        transform_img = transforms.Compose([
            transforms.Resize((400, 400)),  # should actually be 1:3 but broke the system
            transforms.ToTensor()])

        # create the training DataLoader
        print(os.listdir(self.filepath+"train/"))
        train_data = ImageFolderWithPaths(root=self.filepath + "train/", transform=transform_img)
        train_data_loader = data.DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)
        # create the testing DataLoader
        test_data = ImageFolderWithPaths(root=self.filepath + "validation/", transform=transform_img)
        test_data_loader = data.DataLoader(test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True)

        # print useful information about the data
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx)
        return train_data_loader, test_data_loader

    def train(self):
        """
        Trains the model until the accuracy plateaus or the maximum
        number of epochs to train has been reached. Logs the results of the training after each epoch.
        """
        # Training
        accuracy = []
        self.epoch = 0
        diff_avg = 100
        loss = 0
        # while the accuracy is decreasing or the number of epochs is <15, and while the number of epochs <= 100
        while (diff_avg >= 0.05 or self.epoch <= 15) and self.epoch <= 100:
            self.epoch += 1
            loss_out = 0
            # runs everything in the training DataLoader through the model
            for i, (image, label, path) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                outputs = self.model(image)
                loss = self.loss_func(outputs, label)
                loss_out += loss
                loss.backward()
                self.optimizer.step()

            # tests if the training results match the labels in the DataLoader
            training_accuracy = self.test(self.train_data_loader)
            testing_accuracy = self.test(self.test_data_loader)
            # get data analysis info and save it to a .jsonl file
            confusion_matrix, precision, recall, incorrect_paths = self.cf_matrix(self.test_data_loader)
            with open(self.log_filepath, 'a') as f:
                print("epoch", self.epoch)
                f.write(json.dumps(
                    [self.epoch, loss.item(), training_accuracy, testing_accuracy, precision, recall, incorrect_paths, confusion_matrix.tolist()]) + "\n")

                # Calculate the average increase in accuracy over the last 5 epochs to determine if training
                # should continue.
                if len(accuracy) == 5:
                    accuracy.pop(0)
                accuracy.append(loss.item())
                if len(accuracy) >= 2:
                    diff_sum = 0
                    for i in range(1, len(accuracy)):
                        diff_sum += abs(accuracy[i] - accuracy[i - 1])
                    diff_avg = abs(diff_sum / (len(accuracy) - 1))

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
                for test_step, data in enumerate(data_loader):
                    test_images, labels, path = data
                    outputs = self.model(test_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        # calculate and return percent accuracy
        return 100 * correct / total

    # return the confusion matrix, followed by precision and recall
    def cf_matrix(self, data_loader):
        """
        Performs preliminary data analysis of the testing DataLoader.

        :param data_loader: the test_data_loader to gather information from
        :type data_loader: DataLoader
        :returns: a confusion matrix, precision and recall values, and the paths of misclassified samples
        """
        list_predicted = []
        list_labels = []
        incorrect_paths = [[[] for i in range(20)] for j in range(20)]
        
        precision = []
        recall = []
        
        with torch.no_grad():
            for test_step, data in enumerate(data_loader):
                # gather data from the DataLoader
                test_images, labels, path = data
                model_output = self.model(test_images)
                _, predicted = torch.max(model_output.data, 1)
                list_predicted.extend(list(predicted.numpy()))
                list_labels.extend(list(labels.numpy()))
                is_correct = (predicted == labels).sum().item()
                # add misclassified sample paths to a list
                for x in range(len(labels) - 1):
                    if predicted[x] != labels[x]:
                        path[x] = path[x].split('/')[len(path[x].split('/')) - 1]
                        incorrect_paths[predicted[x]][labels[x]].append(path[x])
<<<<<<< HEAD
        
=======

        # create a confusion matrix
>>>>>>> 1c2a343739f2c294b0870510c1633c506aee7eb2
        confusion = np.zeros((self.model.num_class, self.model.num_class))
        for pred, truth in zip(list_predicted, list_labels):
            confusion[pred][truth] += 1

        # calculate precision and recall
        for i in range (0, len(confusion)):
            if sum(confusion[i]) != 0:
                precision.append(confusion[i][i] / sum(confusion[i]))
            else:
                precision.append(0.0)
            recall.append(confusion[i][i] / sum(confusion[:,i]))
        
        return confusion, precision, recall, incorrect_paths

    def configure_log(self):
        """
        Configures a .jsonl file with the correct column headers.
        """
        with open(self.log_filepath, 'w') as f:
            f.write(json.dumps(['epoch', 'loss', 'training_accuracy', 'testing_accuracy', 'precision', 'recall', 'incorrect_paths', 'confusion_matrix']) + "\n")


if __name__ == '__main__':
    """
    Trains a single instance of the model with the parameters specified by the constants in the file header.
    """
    if len(sys.argv) > 1:
        birdsong = Training(sys.argv[1], EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)
    else:
        birdsong = Training(FILEPATH, EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)

    birdsong.train()
