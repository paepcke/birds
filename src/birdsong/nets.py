import copy
import socket

from torch import device
from torch import hub, cuda
import torch
from torchvision.models.resnet import BasicBlock

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Class NetUtils ----------------
class NetUtils:
    
    #------------------------------------
    # get_net
    #-------------------
    
    @classmethod
    def get_net(cls, net_name, **kwargs):
        if net_name.lower() == 'basicnet':
            return BasicNet(**kwargs)
        elif net_name.lower() == 'resnet':
            return cls.get_resnet_partially_trained(**kwargs)
        else:
            raise NotImplementedError(f"Network {net_name} unavailable")

    #------------------------------------
    # get_resnet_partially_trained 
    #-------------------

    @classmethod
    def get_resnet_partially_trained(cls, 
                                     num_classes=None, 
                                     num_layers_to_retain=6,
                                     resnet_version=18,
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
        @param resnet_version: which Resnet to return: 18 or 50
        @type resnet_version: int
        @return: a fresh model
        @rtype: pytorch.nn 
        '''

        if num_classes is None:
            raise ValueError("Resnetxx requires a num_classes argument")
        if resnet_version not in (18,50):
            raise ValueError("Resnet version must be 18 or 50")
        
        model = hub.load('pytorch/vision:v0.6.0', 
                         'resnet18' if resnet_version == 18 else 'resnet50', 
                         pretrained=True if num_layers_to_retain > 0 else False
                         )

        if to_grayscale:
            model = cls._first_layer_to_in_channel1(model, resnet_version)

        cls.freeze_model_layers(model, num_layers_to_retain)

        num_in_features = model.fc.in_features
        
        model.fc = nn.Linear(num_in_features, num_classes)
        
        # Create a property on the model that 
        # returns the number of output classes:
        model.num_classes = model.fc.out_features
        return model
    
    #------------------------------------
    # _first_layer_to_in_channel1
    #-------------------

    @classmethod
    def _first_layer_to_in_channel1(cls, model, resnet_version):
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
        
        @param cls:
        @type cls:
        @param model:
        @type model:
        @param resnet_version:
        @type resnet_version:
        '''
        
        # One could get the first layer's block0
        # conv1 layer like this:
        #   list(model.layer1.children())[0].conv1
        # But this is shorter:
        l0_b0_conv1 = model.conv1
        saved_conv1_weight = copy.deepcopy(l0_b0_conv1.weight)
        saved_conv1_bias   = copy.deepcopy(l0_b0_conv1.bias)
        curr_conv_layer_attrs = {
            'kernel_size' : l0_b0_conv1.kernel_size,
            'out_channels' : l0_b0_conv1.out_channels,
            'stride' : l0_b0_conv1.stride,
            'padding' : l0_b0_conv1.padding,
            'bias' : l0_b0_conv1.bias
            }
        # Create a Conv2d layer just like the existing one, 
        # but change the input channel (from its default 64) to 1:
        new_l0_b0_conv1 = nn.Conv2d(1, **curr_conv_layer_attrs)
        
        # The original conv1 weights contain three channels,
        # for RG&B. Average them to collapse into one input
        # channel. We start with:
        #
        #     saved_conv1_weight.shape 
        #        --> torch.Size([64, 3, 7, 7])
        #
        # and want to end up with: torch.Size([64, 1, 7, 7])
        # The keepdim=True is required, b/c otherwise the mean
        # method removes dim1 from the result:
        
        grayscale_weight = torch.mean(saved_conv1_weight, 
                                      dim=1, 
                                      keepdim=True)
        # ***** Confused about Parameter vs. its data
        saved_conv1_weight.data = grayscale_weight
        
        new_l0_b0_conv1.weight.data = saved_conv1_weight
        new_l0_b0_conv1.bias        = saved_conv1_bias
        
        model.conv1 = new_l0_b0_conv1
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
