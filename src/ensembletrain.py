import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.multiprocessing
import logging
from datetime import datetime, date, time
from nets import BasicNet
import json

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
FILEPATH = "/home/data/birds/ENSEMBLE_DATA/"

EPOCHS = 6
SEED = 42
BATCH_SIZE = 32
KERNEL_SIZE = 7
NET_NAME = 'BasicNet'
GPU = 0


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        super(datasets.ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                   transform=transform, target_transform=target_transform,
                                                   is_valid_file=is_valid_file)

        self.imgs = self.samples

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

#Primary training class. Creates one net, and trains it. Contains methods for outputting 
#confusion matrix, accuracy, and other stats, and for logging them. 
class Training:
    def __init__(self, file_path, train_path, epochs, batch_size, kernel_size, seed=42, net_name='BasicNet', gpu=0):
        # handle seed
        if seed is not None:
            self.set_seed(seed)

        # define device used
        #at some point may be worth putting different nets on different parts of cuda for speed
        self.device = torch.device("cuda:" + str(gpu) if (torch.cuda.is_available() and gpu is not None) else "cpu")
        print(self.device)

        # variables
        self.EPOCHS = epochs
        self.epoch = 0
        self.BATCH_SIZE = batch_size
        self.KERNEL_SIZE = kernel_size
        self.filepath = file_path
        self.trainpath = train_path

        # configure log json file
        now = datetime.now()
        self.log_filepath = now.strftime("%d-%m-%Y") + '_' + now.strftime("%H-%M") + "_K" + str(
            self.KERNEL_SIZE) + '_B' + str(self.BATCH_SIZE) + '.jsonl'
        self.configure_log()

        # configure net
        self.train_data_loader, self.test_data_loader, self.num_classes = self.import_data()
        self.model = self.get_net(net_name, self.num_classes, self.BATCH_SIZE, self.KERNEL_SIZE, GPU)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        #lambda1 = lambda epoch: 0.01 * (1.00 ** self.epoch)
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1, last_epoch=-1)
        self.loss_func = nn.CrossEntropyLoss()

    #Not currently in use, relevant if we want to test different net architectures
    def get_net(self, net_name, num_class, batch_size, kernel_size, gpu):
        if net_name == 'BasicNet':
            return BasicNet(num_class, batch_size, kernel_size, gpu)
        elif NET_NAME == 'Resnet18':  # change this to use Andreas's Resnet
            return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        # default to basic net
        else:
            raise ValueError("Unknown net name")

    # set the seed across all different necessary platforms
    # to allow for comparison of different model seeding.
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do! leave them in, I guess? 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # access data
    # returns the training dataloader, testing dataloader, and number of classes (to be passed as parameter to net)
    def import_data(self):
        # define the transform to be done on all images -- converts to tensor and resizes
        transform_img = transforms.Compose([
            transforms.Resize((400, 400)),  # should actually be 1:3 but broke the system
            transforms.ToTensor()])

        train_data = ImageFolderWithPaths(root=self.filepath + self.trainpath, transform=transform_img)
        train_data_loader = data.DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                            pin_memory=True)

        test_data = ImageFolderWithPaths(root=self.filepath + "validation/", transform=transform_img)
        test_data_loader = data.DataLoader(test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True)

        
        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx)
        num_classes = len(train_data.class_to_idx)
        return train_data_loader, test_data_loader, num_classes

    #main train loop
    #also calls testing
    def train(self):
        # Training
        accuracy = []
        self.epoch = 0
        diff_avg = 100
        loss = 0
        while (diff_avg >= 0.03 or self.epoch <= 15) and self.epoch <= 65: 
            self.epoch += 1
            loss_out = 0
            for i, (image, label, path) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                outputs = self.model(image)
                loss = self.loss_func(outputs, label)
                loss_out += loss
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()

            training_accuracy = self.test(self.train_data_loader)
            #there are errors with confusion matrix calculation on first epoch sometimes
            try:
                confusion_matrix, precision, recall, incorrect_paths = self.cf_matrix(self.train_data_loader)
            except: 
                print("error with confusion matrix generation")
                confusion_matrix = None
                precision = None    
                recall = None

            with open(self.log_filepath, 'a') as f:
                print("epoch", self.epoch)
                f.write(json.dumps(
                    [self.epoch, self.trainpath, loss.item(), training_accuracy, precision, recall, confusion_matrix.tolist()]) + "\n")

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
        correct = 0
        total = 0
        with torch.no_grad():
            if len(list(enumerate(data_loader))) == 3:
                for test_step, data in enumerate(data_loader):
                    test_images, labels = data
                    outputs = self.model(test_images)
                    print(outputs)
                    print(type(outputs))
                    print(outputs.size())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            else:
                for test_step, data in enumerate(data_loader):
                    test_images, labels, path = data
                    outputs = self.model(test_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # return the confusion matrix, precision, and recall
    def cf_matrix(self, data_loader):
        list_predicted = []
        list_labels = []
        incorrect_paths = [[""]*20]*20
        
        precision = []
        recall = []
        
        with torch.no_grad():
            for test_step, data in enumerate(data_loader):
                test_images, labels, path = data
                model_output = self.model(test_images)
                _, predicted = torch.max(model_output.data, 1)
                list_predicted.extend(list(predicted.numpy()))
                list_labels.extend(list(labels.numpy()))
                is_correct = (predicted == labels).sum().item()
                for x in range(len(labels) - 1):
                    if predicted[x] != labels[x]:
                        path[x] = path[x].split('/')[len(path[x].split('/')) - 1]
                        incorrect_paths[predicted[x]][labels[x]] += (path[x] + ",")
        
        confusion = np.zeros((self.model.num_class, self.model.num_class))
        for pred, truth in zip(list_predicted, list_labels):
            confusion[pred][truth] += 1
            
        for i in range (0, len(confusion)):
            if sum(confusion[i]) != 0:
                precision.append(confusion[i][i] / sum(confusion[i]))
            else:
                precision.append(0.0)
            recall.append(confusion[i][i] / sum(confusion[:,i]))
        
        return confusion, precision, recall, incorrect_paths

    def configure_log(self):
        with open(self.log_filepath, 'w') as f:
            f.write(json.dumps(['epoch', 'netnumber' 'loss', 'training_accuracy', 'precision', 'recall', 'confusion_matrix']) + "\n")

#Class that creates multiple Training objects
class Ensemble:
    def __init__(self):
        self.trainings = []
        #list of folders with training data, under parent directory FILEPATH defined at top of file
        self.folders = ['train1/', 'train2/', 'train3/', 'train4/']

    #create and instantiate a Training object (ie. a net) for each othe folders in self.folders
    def train_models(self):
        kernel = 7
        batch = 64
        
        for folder in self.folders:
            #FILEPATH is parent s defined at top of file
            self.trainings.append(Training(FILEPATH, folder, EPOCHS, batch, kernel, SEED, NET_NAME, GPU))

        #train the nets
        for training in self.trainings:
            training.train()

        self.test_output()

    #test output of the four nets by combining + testing on testing dataset
    #I'm still sketched out by this whole function, I want to test it more rigorously 
    #to ensure it's actually doing what's expected
    def test_output(self):
        correct = 0
        total = 0
        #all of the Training objects have a test dataloader, this grabs it from the first one
        data_loader = self.trainings[0].test_data_loader

        with torch.no_grad():
            for test_step, data in enumerate(data_loader):
                
                test_images, labels, path = data
                outputs = []
                output_data = []
                #gets output of each net
                for training in self.trainings:
                    outputs.append(training.model(test_images).data)

                #concatenates the outputs, then calls torch.max across each row
                output_data = torch.cat((outputs[0], outputs[1], outputs[2], outputs[3]), 1)
                _, predicted = torch.max(output_data, 1)

                #calculate accuracy

                #mappings are manually entered, as they depend on the human decision of how to sort the dataset
                #the testing dataset goes from 0-13 in alphabetical order
                #the training datasets go in order (train1, train2, etc.) and alphabetical order within that
                #I know this is cursed, someone fix this in the future lol
                mappings = {0:6, 1:8, 2:9, 3:11, 4:1, 5:2, 6:5, 7:10, 8:3, 9:4, 10:13, 11:0, 12:7, 13:12}
                for i in range(0, list(predicted.size())[0]):
                    predicted[i] = torch.tensor(mappings[predicted[i].item()])

                    
                #dump data to a new file
                fileout = open("NEWEST_ENSEMBLE_OUTPUT.txt","a")
                fileout.write("predicted is" + str(predicted))
                fileout.write("actual is" + str(labels))
                fileout.close()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("accuracy here")
        print(100 * correct / total)
        fileout = open("NEWEST_ENSEMBLE_OUTPUT.txt","a")
        fileout.write("final accuracy is" + str(100 * correct / total))
        fileout.close()


    #currently untested, might be broken
    #runs through a variety of kernel/batch sizes
    def test_bed():
        kernel_upper = 9
        kernel_lower = 7
        batch_upper = 128
        batch_lower = 1

        ks = kernel_lower
        while ks <= kernel_upper:
            bs = batch_lower
            while bs <= batch_upper:
                for folder in self.folders:
                    self.trainings.append(Training(FILEPATH, folder, EPOCHS, batch, kernel, SEED, NET_NAME, GPU))
                for model in self.trainings:
                    model.train()

                bs *= 2
            ks += 2


if __name__ == '__main__':
    birdsong = Ensemble()
    birdsong.train_models()

