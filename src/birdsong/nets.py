import socket

from torch import device
from torch import hub, cuda
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Class NetUtils ----------------
class NetUtils:
    
    #------------------------------------
    # get_resnet_partially_trained 
    #-------------------

    @classmethod
    def get_resnet_partially_trained(cls, 
                                       num_classes, 
                                       num_layers_to_retain=6,
                                       resnet_version=18
                                       ): 

        '''
        Obtains the pretrained resnet18 model from the Web
        if not cached. Then freezes num_layers_to_retain
        layers to preserve the pre-training. If num_layers_to_retain
        is zero, the untrained version is used.
        
        If running under Distributed Data Processing (DDP) 
        protocol, only the master node will download, and
        then share with the others.
        
        @param num_classes: number of target classes
        @type num_classes: int
        @param num_layers_to_retain: how many layers to
            freeze, protecting them from training.
        @type num_layers_to_retain: int
        @param resnet_version: which Resnet to return: 18 or 50
        @type resnet_version: int
        @return: a fresh model
        @rtype: pytorch.nn 
        '''
        
        if resnet_version not in (18,50):
            raise ValueError("Resnet version must be 18 or 50")
        
        model = hub.load('pytorch/vision:v0.6.0', 
                         'resnet18' if resnet_version == 18 else 'resnet50', 
                         pretrained=True if num_layers_to_retain > 0 else False
                         )

        cls.freeze_model_layers(model, num_layers_to_retain)

        num_in_features = model.fc.in_features
        
        model.fc = nn.Linear(num_in_features, num_classes)
        
        # Create a property on the model that 
        # returns the number of output classes:
        model.num_classes = model.fc.out_features
        return model

    #------------------------------------
    # get_model_ddp 
    #-------------------

    @classmethod
    def get_model_ddp(cls, 
                      rank, 
                      local_leader_rank, 
                      log,
                      resnet_version,
                      num_layers_to_retain
                      ):  # @DontTrace
        '''
        Determine whether this process is the
        master node. If so, obtain the pretrained
        resnet18 model. Then distributed the model
        to the other nodes. 
        
        @param rank: this process' rank
            in the distributed data processing sense
        @type rank: int
        @param local_leader_rank: the lowest rank on this machine
        @type local_leader_rank: int
        @param log: logging service to log to
        @type log: LoggingService
        '''

        if resnet_version not in (18,50):
            raise ValueError("Resnet version must be 18 or 50")

        hostname = socket.gethostname()
        # Let the local leader download
        # the model from the Internet,
        # in case it is not already cached
        # locally:
        
        # Case 1: not on a GPU machine:
        device = device('cuda' if cuda.is_available() else 'cpu')
        if device == device('cpu'):
            model = hub.load('pytorch/vision:v0.6.0', 
                             'resnet18' if resnet_version == 18 else 'resnet50', 
                             pretrained=True if num_layers_to_retain > 0 else False
                             )
            
        # Case2a: GPU machine, and this is this machine's 
        #         leader process. So it is reponsible for
        #         downloading the model if it is not cached:
        elif rank == local_leader_rank:
            log.info(f"Procss with rank {rank} on {hostname} loading model")
            model = hub.load('pytorch/vision:v0.6.0', 
                             'resnet18' if resnet_version == 18 else 'resnet50', 
                             pretrained=True if num_layers_to_retain > 0 else False
                             )

            # Allow the others on this machine
            # to load the model (guaranteed to 
            # be locally cached now):
            log.info(f"Procss with rank {rank} on {hostname} waiting for others to laod model")
            dist.barrier()
        # Case 2b: GPU machine, but not the local leader. Just
        #          wait for the local leader to be done downloading:
        else:
            # Wait for leader to download the
            # model for everyone on this machine:
            log.info(f"Process with rank {rank} on {hostname} waiting for leader to laod model")
            dist.barrier()
            # Get the cached version:
            log.info(f"Procss with rank {rank} on {hostname} laoding model")
            model = hub.load('pytorch/vision:v0.6.0', 
                             'resnet18' if resnet_version == 18 else 'resnet50', 
                             pretrained=True if num_layers_to_retain > 0 else False
                             )

        model = cls.freeze_model_layers(model, num_layers_to_retain)

        return model
    
    #------------------------------------
    # freeze_model_layers 
    #-------------------
    
    @classmethod
    def freeze_model_layers(cls, model, num_layers_to_retain):
        '''
        Given a model, freeze as num_layers_to_retain layers,
        and return the model. Freezing a layer sets the required_grad
        attribute of the layer's weight tensors to False
        
        @param cls:
        @type cls:
        @param model: model to partially freeze
        @type model: pytorch.nn
        @param num_layers_to_retain: how many layers to freeze
        @type num_layers_to_retain: int
        @requires: modified model
        @rtype: pytorch.nn
        '''

        if num_layers_to_retain == 0:
            return model
        
        layers_frozen = 0
        while layers_frozen < num_layers_to_retain:
            # Freeze the bottom num_layers_to_retain layers:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = False
            layers_frozen += 1
        return model
        

# ---------------------- BasicNet --------------

class BasicNet(nn.Module):
    def __init__(self, 
                 num_class, 
                 batch_size=32, 
                 kernel_size=5, 
                 processor=None):
        super(BasicNet, self).__init__()
        self.gpu = processor
        self.bs = batch_size
        self.ks = kernel_size
        self.num_class = num_class
        self.conv1 = nn.Conv2d(3, 6, self.ks)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.bs, self.ks)
        print("batch size: " + str(self.bs))
        print("kernel size: " + str(self.ks))
        self.fc1 = nn.Linear(self.bs * int((99 - (self.ks + 1) / 2) ** 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_class)

    def forward(self, x):
        if self.gpu is not None:
            x.cuda(self.gpu)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), self.bs * int((99 - (self.ks + 1) / 2) ** 2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------ Class Resnet18Grayscale --------------

# class Resnet18Grayscale(ResNet):
class Resnet18Grayscale(nn.Module):
    '''
    Not Tested in a Long Time
    
    A Resnet18 variant that accepts single-channel
    grayscale images instead of RGB.
    Using this class saves space from not having
    to replicate our single-layer spectrograms three
    times to pretend they are RGB images.
    '''

    # ------------------------------------
    # Constructor
    # -------------------

    def __init__(self, *args, **kwargs):
        '''
        Args and kwargs as per https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        class ResNet.__init__()
        '''
        # The [2,2,2,2] is an instruction to the
        # superclass' __init__() for how many layers
        # of each type to create. This info makes the
        # ResNet into a ResNet18:
        self.num_class = 10
        super().__init__(BasicBlock, [2, 2, 2, 2], *args, **kwargs)

        # Change expected channels from 3 to 1
        # The superclass created first layer
        # with the first argument being a 3.
        # We just replace the first layer:
        self.inplanes = 64  # ******* Should be batch size?
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

    # ------------------------------------
    # forward
    # -------------------

    def forward(self, x):
        out_logit = super().forward(x)

        # Since we have binary classification,
        # the Sigmoid function does what a
        # softmax would do for multi-class:

        out_probs = nn.Sigmoid()(out_logit)
        return out_probs

    # ------------------------------------
    # device_residence
    # -------------------

    def device_residence(self):
        '''
        Returns device_residence where model resides.
        Can use like this to move a tensor
        to wherever the model is:
            some_tensor.to(<model_instance>.device_residence())
        '''
        return next(self.parameters()).device
