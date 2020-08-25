import os
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.multiprocessing
import logging
from datetime import datetime, date, time
import nvidia_smi
from nets import BasicNet

FILEPATH = "/home/data/birds/Birdsong_Spectrograms/"
EPOCHS = 60
SEED = 42
BATCH_SIZE = 32
KERNEL_SIZE = 5
NET_NAME = 'BasicNet'
GPU = 0

class Training:
    def __init__(self, FILEPATH, EPOCHS = 60, SEED = 42, BATCH_SIZE = 32, KERNEL_SIZE = 5, NET_NAME = 'BasicNet', GPU = 0):
        #handle seed
        if SEED is not None:
            self.set_seed(SEED)

        #define device used
        self.device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")
        print(self.device)

        #variables
        self.EPOCHS = EPOCHS
        self.FILEPATH = FILEPATH
        self.BATCH_SIZE = BATCH_SIZE
        self.KERNEL_SIZE = KERNEL_SIZE
        self.filepath = FILEPATH
        self.configure_log()

        #configure net
        self.model = self.getNet(NET_NAME, self.BATCH_SIZE, self.KERNEL_SIZE, GPU)
        self.train_data_loader, self.test_data_loader = self.importData()        
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)
        self.loss_func = nn.CrossEntropyLoss()
        logging.info("batch size: " + str(self.BATCH_SIZE))
        logging.info("kernel size: " + str(self.KERNEL_SIZE))


    def getNet(self, NET_NAME, BATCH_SIZE, KERNEL_SIZE, GPU):
        if (NET_NAME == 'BasicNet'):
            return BasicNet(BATCH_SIZE, KERNEL_SIZE, GPU)
        if (NET_NAME == 'Resnet18'):  #change this to use Andreas's Resnet
            return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained = False)
        #default to basic net
        else:
            return BasicNet(BATCH_SIZE, KERNEL_SIZE, GPU) 


    #set the seed across all different necessary platforms
    #to allow for comparison of different model seeding.    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    #access data
    def importData(self):
        #define the transform to be done on all images
        transform_img = transforms.Compose([
            transforms.Resize((400, 400)),   #should actually be 1:3 but broke the system
            transforms.ToTensor()])

        train_data = torchvision.datasets.ImageFolder(root = self.filepath + "train/", transform = transform_img)
        train_data_loader = data.DataLoader(train_data, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)

        test_data = torchvision.datasets.ImageFolder(root = self.filepath + "validation/", transform = transform_img)
        test_data_loader  = data.DataLoader(test_data, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True) 

        print("Number of train samples: ", len(train_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx) 
        return (train_data_loader, test_data_loader)

    def train(self):
        # Training
        accuracy = []
        epoch = 0
        diff_avg = 100
        while diff_avg >= 0.75 or epoch <= 15:
        # for epoch in range(self.EPOCHS):
            epoch +=1
            loss_out = 0
            for i, (image, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()  
                outputs = self.model(image)
                loss = self.loss_func(outputs, label)   
                loss_out += loss   
                loss.backward() 
                self.optimizer.step()
            
            # log every 3 epochs
            if (epoch%3 == 0):
                training_accuracy = self.test(self.train_data_loader)
                testing_accuracy = self.test(self.test_data_loader)
                confusion_matrix = self.cf_matrix(self.test_data_loader)
                with open(self.LOG_FILEPATH, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([epoch, loss, training_accuracy, testing_accuracy, confusion_matrix])
                if len(accuracy) == 5:
                    accuracy.pop(0)
                accuracy.append(testing_accuracy)
                if len(accuracy) >= 2:
                    diff_sum = 0
                    for i in range(1, len(accuracy)):
                        diff_sum += accuracy[i] - accuracy[i-1]
                    diff_avg = diff_sum / (len(accuracy) -1)
                    print("accuracy[] is: " + str(accuracy))
                    print("epoch is: " + str(epoch))
                    print("accuracy is: " + str(testing_accuracy))
                    print("diff_avg is: " + str(diff_avg))
    
    #return percent accuracy
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
        return (100 * correct / total)

    #return the confusion matrix
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
            confusion[pred][truth] +=1
        return confusion


    def configure_log(self):
        now = datetime.now()
        self.LOG_FILEPATH = now.strftime("%d-%m-%Y") + '_' + now.strftime("%H-%M") + '.csv'
        with open(self.LOG_FILEPATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['epoch', 'loss', 'training_accuracy', 'testing_accuracy', 'confusion_matrix'])

def test_bed():
    kernel_upper = 11
    kernel_lower = 3
    batch_upper = 256
    batch_lower = 1

    ks = kernel_lower
    while ks <= kernel_upper:
        bs = batch_lower
        while bs <= batch_upper:
            logging.info("testing initialized")
            logging.info("batch size: " + str(bs))
            logging.info("kernel size: " + str(ks))
            birdsong = Training(FILEPATH, EPOCHS, SEED, bs, ks, NET_NAME, GPU)
            birdsong.train()
            bs *= 2
        ks += 2
    loggging.info("all configurations tested")
                
if __name__ == '__main__':
    test_bed()
    #birdsong = Training(FILEPATH, EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE, NET_NAME, GPU)
    #birdsong.train()



    
    
    
