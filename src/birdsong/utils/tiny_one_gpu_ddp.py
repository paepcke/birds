#!/usr/bin/env python

import os
import sys
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

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

class MinimalDDP:
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

        # For saving model copies
        # before and after back prop
        # for each loop iteration:
        
        before = []
        after  = []
        
        for _epoch in range(self.epochs):
            for _i in range(self.samples):
                
                optimizer.zero_grad()
                outputs = ddp_model(torch.randn(20, 10).to(rank))
                labels = torch.randn(20, 5).to(rank)
                
                # Copy and save model copies before and
                # after back prop:
                before_model = ddp_model.cpu()
                before.append(copy.deepcopy(before_model.state_dict()))
                ddp_model.cuda(0)
                
                loss_fn(outputs, labels).backward()

                optimizer.step()
                
                after_model = ddp_model.cpu()
                after.append(copy.deepcopy(after_model.state_dict()))
                ddp_model.cuda(0)
                
                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        dist.barrier()

        # Save the state_dirs of all before-prop
        # and after-prop model copies; each in its
        # own file:
        self.save_model_arrs(before, after, model_save_dir)
        
        self.cleanup()

    #------------------------------------
    # save_model_arrs 
    #-------------------
    
    def save_model_arrs(self, before_arr, after_arr, model_save_dir):
        '''Save state_dict of modesl in arrays to files'''
        
        print(f"Proc{rank}: saving arrays of before and after models.")
        
        for i, (before_state, after_state) in enumerate(zip(before_arr, after_arr)):
            msg = f"Loop {i}: before/after states are "
            if before_state.eq(after_state):
                msg += "equal"
            else:
                msg += "different"

    #------------------------------------
    # cleanup 
    #-------------------

    def cleanup(self):
        dist.destroy_process_group()
        print(f"Rank {rank} is done.")
        
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
    model_save_dir = '/tmp'
    min_ddp = MinimalDDP()
    min_ddp.demo_basic(rank, world_size, model_save_dir)
