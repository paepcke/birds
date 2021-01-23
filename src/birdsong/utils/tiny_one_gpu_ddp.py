#!/usr/bin/env python

import os
import sys
import copy

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch import randn

from torch.nn.parallel import DistributedDataParallel as DDP

#*****************
#
import socket
if socket.gethostname() in ('quintus', 'quatro'):
    # Point to where the pydev server
    # software is installed on the remote
    # machine:
    sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))

    import pydevd
    global pydevd
    # Uncomment the following if you
    # want to break right on entry of
    # this module. But you can instead just
    # set normal Eclipse breakpoints:
    #*************
    print("About to call settrace()")
    #*************
    pydevd.settrace('localhost', port=4040)
#****************

class TinyDDP:
    '''Test whether DDP really does something'''
    
    epochs  = 2
    samples = 3

    #------------------------------------
    # setup
    #-------------------

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    #------------------------------------
    # demo_basic
    #-------------------

    def demo_basic(self, rank, world_size, model_save_dir='/tmp'):
        '''The action: train model; save intermediate states'''
            
        print(f"Running basic DDP example on rank {rank}.")
        self.setup(rank, world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        for _epoch in range(self.epochs):
            for _i in range(self.samples):
                
                optimizer.zero_grad()
                outputs = ddp_model(randn(20, 10).to(rank))
                labels = randn(20, 5).to(rank)
                
                before_model = ddp_model.cpu()
                before_state = copy.deepcopy(before_model.state_dict())
                ddp_model.to(rank)

                loss_fn(outputs, labels).backward()
                optimizer.step()
                
                after_model = ddp_model.cpu()
                after_state = after_model.state_dict()
                states_equal = True
                for before_parm, after_parm in zip(before_state.values(),
                                                   after_state.values()):
                    if before_parm.ne(after_parm).any():
                        states_equal = False
                ddp_model.to(rank)
                print(f"Before and after are {('equal' if states_equal else 'different')}")
                
                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        self.cleanup()

    #------------------------------------
    # cleanup 
    #-------------------

    def cleanup(self):
        dist.destroy_process_group()
        print(f"Done.")
        
# ------------------------ Toy Model ----------

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# ------------------------ Main ------------
if __name__ == '__main__':

    rank           = 0
    world_size     = 1
    min_ddp = TinyDDP().demo_basic(rank, world_size)
