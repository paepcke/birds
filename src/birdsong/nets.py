import copy
import socket
import re

from torch import device
from torch import hub, cuda
import torch

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Class NetUtils ----------------
class NetUtils:
    '''
    Creating neural networks. Main entry point
    is method get_net(). Serves out a basic network
    BasicNet, or a version of resnet (18, 34, 50).
    When requesting resnets, caller can specify a nunmber
    of layers to have pretrained. Thus a fully, or
    partially pretrained resnet model can be obtained. 
    
    '''
    
    # Regex pattern to separate 'resnet18' into ('resnet', '18'),
    # or 'resnet' without version into ('resnet', ''):
     
    net_name_separation_pat = re.compile(r'([a-zA-Z]*)([0-9]*)$')
    
    resnet_pattern = re.compile(r'resnet([0-9]*)$')
    
    #------------------------------------
    # get_net
    #-------------------
    
    @classmethod
    def get_net(cls, net_name, **kwargs):
        
        (net_name, net_version) = cls.name_and_version_from_net_name(net_name)
        kwargs['net_version'] = net_version
        net_name = net_name.lower()

        if net_name == 'basicnet':
            return BasicNet(**kwargs)
        elif net_name == 'resnet':
            return cls._get_resnet_partially_trained(**kwargs)
        elif net_name == 'densenet':
            return cls._get_densenet_partially_trained(**kwargs)
        else:
            raise NotImplementedError(f"Network {net_name} unavailable")

    #------------------------------------
    # _get_resnet_partially_trained 
    #-------------------

    @classmethod
    def _get_resnet_partially_trained(cls, 
                                     num_classes=None, 
                                     num_layers_to_retain=6,
                                     net_version=18,
                                     to_grayscale=False
                                     ): 

        '''
        Obtains the pretrained resnet18 model from the Web
        if not cached. Then:
           o Freezes the leftmost num_layers_to_retain layers
             so that they are unaffected by subsequent training.
             
           o Modifies the number of output classes from resnet's
             defaulat 1000 to num_classes.
             
           o Modifies the input layer to expect grayscale,
             i.e. only one channel, instead of three. The
             weights are retained from the pretrained model.
              
        If num_layers_to_retain is zero, the untrained version 
        is used.
        
        If running under Distributed Data Processing (DDP) 
        protocol, only the master node will download, and
        then share with the others.
        
        @param num_classes: number of target classes
        @type num_classes: int
        @param num_layers_to_retain: how many layers to
            freeze, protecting them from training.
        @type num_layers_to_retain: int
        @param net_version: which Resnet to return: 18 or 50
        @type net_version: int
        @return: a fresh model
        @rtype: pytorch.nn 
        '''

        
        if num_classes is None:
            raise ValueError("Num_classes argument is required")
        
        net_name = 'resnet'
        available_versions = (18,34,50)
        
        if net_version not in available_versions:
            raise ValueError(f"{net_name} version must be one of {available_versions}")
        
        model = hub.load('pytorch/vision:v0.6.0', 
                         f'{net_name}{net_version}',
                         pretrained=True if num_layers_to_retain > 0 else False
                         )

        if to_grayscale:
            model = cls._first_layer_to_in_channel1(model, net_name)

        cls.freeze_model_layers(model, num_layers_to_retain)

        num_in_features = model.fc.in_features
        
        model.fc = nn.Linear(num_in_features, num_classes)
        
        # Create a property on the model that 
        # returns the number of output classes:
        model.num_classes = model.fc.out_features
        return model
    
    #------------------------------------
    # _get_densenet_partially_trained 
    #-------------------

    @classmethod
    def _get_densenet_partially_trained(cls,
                                        num_classes=None, 
                                        num_layers_to_retain=6,
                                        net_version=161,
                                        to_grayscale=False
                                        ):

        if num_classes is None:
            raise ValueError("Num_classes argument is required")
        
        net_name = 'densenet'
        available_versions = (121,161,169,201)
        
        if net_version not in available_versions:
            raise ValueError(f"{net_name} version must be one of {available_versions}")
        
        model = hub.load('pytorch/vision:v0.6.0', 
                         f'{net_name}{net_version}',
                         pretrained=True if num_layers_to_retain > 0 else False
                         )

        if to_grayscale:
            model = cls._first_layer_to_in_channel1(model, net_name)

        cls.freeze_model_layers(model, num_layers_to_retain)

        # Reduce from 1000 to num_classes targets:
        model = cls._mod_classifier_num_classes(model, 
                                                net_name, 
                                                num_classes)
        
        # Create a property on the model that 
        # returns the number of output classes:
        model.num_classes = num_classes
        return model

    
    #------------------------------------
    # _first_layer_to_in_channel1
    #-------------------

    @classmethod
    def _first_layer_to_in_channel1(cls, model, net_name):
        '''
        Replace input layer with an equivalent
        layer that expects only one channel
        (grayscale), rather than the default 
        three (RGB). There are differences in how
        the network families call their layers.
        So, need to distinguish:
                
        @param cls:
        @type cls:
        @param model:
        @type model:
        @param net_name:
        @type net_name:
        '''
        
        if net_name == 'resnet':
            return cls._resnet_replace_first_layer(model)
        elif net_name == 'densenet':
            return cls._densenet_replace_first_layer(model)

    #------------------------------------
    # _densenet_replace_first_layer 
    #-------------------
    
    @classmethod
    def _densenet_replace_first_layer(cls, model):
        
        
        # Get the input layer module, which
        # is a torch.Conv2d instance:
        
        l_in = model.features.conv0
        
        # l_in.weight.shape is: torch.Size([96, 3, 7, 7])
        # Get the sum of the RGB planes (the dim that has 3):
        grayscale_weight = torch.sum(l_in.weight,
                                     dim=1, 
                                     keepdim=True)

        # Now: grayscale_weight.shape: 
        #    torch.Size([96, 1, 7, 7])
        
        # Make a new input layer with the
        # same parameters as the current one:
        l_in_new = nn.Conv2d(
                1, # in-channel
                l_in.out_channels,
                kernel_size=l_in.kernel_size,
                stride=l_in.stride,
                padding=l_in.padding,
                dilation=l_in.dilation,
                groups=l_in.groups,
                bias=l_in.bias,
                padding_mode=l_in.padding_mode
                )

        
        # Suppress tracking of the dimension
        # change by the layer; therefore the
        # no_grad():
        with torch.no_grad():
            del l_in_new.weight
            l_in_new.weight = nn.Parameter(grayscale_weight)

        model.features.conv0 = l_in_new
        
        return model
    
    #------------------------------------
    # _resnet_replace_first_layer 
    #-------------------
    
    @classmethod
    def _resnet_replace_first_layer(cls, model):
        '''
        For resnet18, the first layer has
        two blocks:
        
        model.layer1
        Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        
        @param model:
        @type model:
        '''
        # Get input layer whose weight attr is torch.Size([64, 3, 7, 7])
        l_in = model.conv1
        
        # Weights are: l_in.weight.shape
        # torch.Size([64, 3, 7, 7])
        # Get the sum of the RGB planes (the dim that has 3):
        grayscale_weight = torch.sum(l_in.weight,
                                     dim=1, 
                                     keepdim=True)
        # Now: grayscale_weight.shape: 
        #    torch.Size([64, 1, 7, 7])
        
        # Make a new input layer with the
        # same parameters as the current one:
        l_in_new = nn.Conv2d(
                1, # in-channel
                l_in.out_channels,
                kernel_size=l_in.kernel_size,
                stride=l_in.stride,
                padding=l_in.padding,
                dilation=l_in.dilation,
                groups=l_in.groups,
                bias=l_in.bias,
                padding_mode=l_in.padding_mode
                )
        
        # l_in_new.weight.shape: 
        # torch.Size([64, 1, 7, 7])        

        # Suppress tracking of the dimension
        # change by the layer; therefore the
        # no_grad():
        with torch.no_grad():
            del l_in_new.weight
            l_in_new.weight = nn.Parameter(grayscale_weight)

        model.conv1 = l_in_new
        
        return model

    #------------------------------------
    # _mod_classifier_num_classes 
    #-------------------
    
    @classmethod
    def _mod_classifier_num_classes(cls, model, net_name, new_num_classes):
        '''
        Replaces the output layer such that
        it outputs only as many class logits
        as we have classes. Models pretrained
        on ImageNet come with 1000 class outputs.
        
        @param model: model instance to modify
        @type model: torchvision.model
        @param net_name: resnet or densenet
        @type net_name: str
        @param new_num_classes: desired number of 
            output classes
        @type new_num_classes: int
        @return: modified model instance
        @rtype: torchvision.model
        '''
        
        if net_name == 'resnet':
            num_in_features = model.fc.in_features
            model.fc = nn.Linear(num_in_features, new_num_classes)
        elif net_name == 'densenet':
            num_in_features = model.classifier.in_features
            model.classifier = nn.Linear(num_in_features, new_num_classes)
        else:
            raise NotImplementedError(f"Model {net_name} not supported")

        return model
    
    #------------------------------------
    # get_model_ddp 
    #-------------------

    @classmethod
    def get_model_ddp(cls, 
                      rank, 
                      local_leader_rank, 
                      log,
                      net_version,
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

        if net_version not in (18,50):
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
                             'resnet18' if net_version == 18 else 'resnet50', 
                             pretrained=True if num_layers_to_retain > 0 else False
                             )
            
        # Case2a: GPU machine, and this is this machine's 
        #         leader process. So it is reponsible for
        #         downloading the model if it is not cached:
        elif rank == local_leader_rank:
            log.info(f"Procss with rank {rank} on {hostname} loading model")
            model = hub.load('pytorch/vision:v0.6.0', 
                             'resnet18' if net_version == 18 else 'resnet50', 
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
                             'resnet18' if net_version == 18 else 'resnet50', 
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
        
    #------------------------------------
    # name_and_version_from_net_name 
    #-------------------
    
    @classmethod
    def name_and_version_from_net_name(cls, net_name_and_version):
        '''
        Given a network name with or without version, 
        return:
           o (<network_name>, <network_version>)
           o None if name cannot possibly be a traditionally
             formatted network name (e.g. starts with a digit)
           o (<network_name>, None) if network name had no
             version.
             
        Ex: returns ('resnet', 18) for input 'resnet18'
            returns ('resnet', None) for input 'resnet'
             
        @param net_name_and_version: name of network, such as 'resnet18'
        @type net_name_and_version: str
        @return: tuple of network name, and version
        @rtype:  {None | (str, {None | int}}
        '''

        match = cls.net_name_separation_pat.match(net_name_and_version)
        if match is None:
            return None
        (net_name, version) = match.groups()
        try:
            version = int(version)
        except ValueError:
            # No version was part of the name
            version = None
        return (net_name, version)


# ---------------------- BasicNet --------------

class BasicNet(nn.Module):
    def __init__(self, 
                 num_classes=None, 
                 batch_size=32, 
                 kernel_size=5, 
                 processor=None):
        
        if num_classes is None:
            raise ValueError("Resnetxx requires a num_classes argument")
        
        super(BasicNet, self).__init__()
        self.gpu = processor
        self.bs = batch_size
        self.ks = kernel_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, self.ks)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.bs, self.ks)
        print("batch size: " + str(self.bs))
        print("kernel size: " + str(self.ks))
        self.fc1 = nn.Linear(self.bs * int((99 - (self.ks + 1) / 2) ** 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

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
