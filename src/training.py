import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.multiprocessing
import logging
from datetime import datetime, date, time
from nets import BasicNet
import json

# FILEPATH = "/Users/amyd/Desktop/Projects/birds/First_Test/"
# FILEPATH = "/home/data/birds/Birdsong_Spectrograms_Augmented/"
# FILEPATH = "/home/data/birds/Birdsong_Spectrograms/"
FILEPATH = "/home/data/birds/NEW_BIRDSONG/"
# FILEPATH = "/Users/LeoGl/PycharmProjects/bird/First_Test/"
EPOCHS = 60
SEED = 42
BATCH_SIZE = 32
KERNEL_SIZE = 7
NET_NAME = 'BasicNet'
# NET_NAME = 'Resnet18'
GPU = 0


class Training:
    def __init__(self, file_path, epochs, batch_size, kernel_size, seed=42, net_name='BasicNet', gpu=0):
        # handle seed
        if seed is not None:
            self.set_seed(seed)

        # define device used
        self.device = torch.device("cuda:" + str(gpu) if (torch.cuda.is_available() and gpu is not None) else "cpu")
        print(self.device)

        # variables
        self.EPOCHS = epochs
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
        if net_name == 'BasicNet':
            return BasicNet(batch_size, kernel_size, gpu)
        if NET_NAME == 'Resnet18':  # change this to use Andreas's Resnet
            return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        # default to basic net
        else:
            return BasicNet(batch_size, kernel_size, gpu)

            # set the seed across all different necessary platforms

    # to allow for comparison of different model seeding.
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # access data
    def import_data(self):
        # define the transform to be done on all images
        transform_img = transforms.Compose([
            transforms.Resize((400, 400)),  # should actually be 1:3 but broke the system
            transforms.ToTensor()])

        print(os.listdir(self.filepath+"train/"))
        train_data = torchvision.datasets.ImageFolder(root=self.filepath + "train/", transform=transform_img)
        train_data_loader = data.DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4,
                                            pin_memory=True)

        test_data = torchvision.datasets.ImageFolder(root=self.filepath + "validation/", transform=transform_img)
        test_data_loader = data.DataLoader(test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4,
                                           pin_memory=True)

        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx)
        return train_data_loader, test_data_loader

    def train(self):
        # Training
        accuracy = []
        epoch = 0
        diff_avg = 100
        loss = 0
        while (diff_avg >= 0.05 or epoch <= 15) and epoch <= 100:
            # for epoch in range(self.EPOCHS):
            epoch += 1
            loss_out = 0
            for i, (image, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                outputs = self.model(image)
                loss = self.loss_func(outputs, label)
                loss_out += loss
                loss.backward()
                self.optimizer.step()

            training_accuracy = self.test(self.train_data_loader)
            testing_accuracy = self.test(self.test_data_loader)
            confusion_matrix = self.cf_matrix(self.test_data_loader)
            with open(self.log_filepath, 'a') as f:
                print("epoch", epoch)
                f.write(json.dumps(
                    [epoch, loss.item(), training_accuracy, testing_accuracy, confusion_matrix.tolist()]) + "\n")

                if len(accuracy) == 5:
                    accuracy.pop(0)
                # accuracy.append(testing_accuracy)
                accuracy.append(loss.item())
                if len(accuracy) >= 2:
                    diff_sum = 0
                    for i in range(1, len(accuracy)):
                        diff_sum += abs(accuracy[i] - accuracy[i - 1])
                    diff_avg = abs(diff_sum / (len(accuracy) - 1))

    # return percent accuracy
    def test(self, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for test_step, data in enumerate(data_loader):
                test_images, labels = data
                outputs = self.model(test_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # return the confusion matrix
    def cf_matrix(self, data_loader):
        list_predicted = []
        list_labels = []
        with torch.no_grad():
            for test_step, data in enumerate(data_loader):
                test_images, labels = data
                model_output = self.model(test_images)
                _, predicted = torch.max(model_output.data, 1)
                list_predicted.extend(list(predicted.numpy()))
                list_labels.extend(list(labels.numpy()))

        confusion = np.zeros((self.model.num_class, self.model.num_class))
        for pred, truth in zip(list_predicted, list_labels):
            confusion[pred][truth] += 1
        return confusion

    def configure_log(self):
        with open(self.log_filepath, 'w') as f:
            f.write(json.dumps(['epoch', 'loss', 'training_accuracy', 'testing_accuracy', 'confusion_matrix']) + "\n")


def test_bed():
    kernel_upper = 9
    kernel_lower = 7
    batch_upper = 128
    batch_lower = 1

    ks = kernel_lower
    while ks <= kernel_upper:
        bs = batch_lower
        while bs <= batch_upper:
            birdsong = Training(FILEPATH, EPOCHS, bs, ks, SEED, NET_NAME, GPU)
            birdsong.train()
            bs *= 2
        ks += 2


if __name__ == '__main__':
    # test_bed()
    birdsong = Training(FILEPATH, EPOCHS, BATCH_SIZE, KERNEL_SIZE, SEED, NET_NAME, GPU)
    birdsong.train()
