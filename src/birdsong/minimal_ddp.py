#!/usr/bin/env python
'''
Created on Jan 19, 2021

@author: paepcke
'''
import os
import sys
import tempfile
import subprocess
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

#*****************
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
# **************** 

class MinimalDDP:

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def demo_basic(self, rank, world_size):
            
        print(f"Running basic DDP example on rank {rank}.")
        self.setup(rank, world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        before = []
        after  = []
        for epoch in range(2):      # 2 Epochs
            for i in range(5):      # 5 Random datapoint/targets
                optimizer.zero_grad()
                outputs = ddp_model(torch.randn(20, 10))
                labels = torch.randn(20, 5).to(rank)
                
                before.append(copy.deepcopy(model))
                loss_fn(outputs, labels).backward()
                after.append(copy.deepcopy(model))

                optimizer.step()
                             
        dist.barrier()

        if rank == 0:
            print("Rank0: saving arrays of before and after models.")
            torch.save([list(_before)
                         for _before in before],
                       '/home/paepcke/tmp/before_models.pth')
            torch.save([list(_after)
                         for _after in after],
                       '/home/paepcke/tmp/after_models.pth')
                             
        self.cleanup()

    def report_model_parm_diffs(self):
        print(f"Saving tensors for rank {rank}")
        for i, one_before in enumerate(before):
            torch.save(one_before, f"/home/paepcke/tmp/PytorchComm/before_rank{rank}_{i}.pth")
    
        for i, one_after in enumerate(after):
            torch.save(one_after, f"/home/paepcke/tmp/PytorchComm/after_rank{rank}_{i}.pth")
        print(f"Done saving tensors for rank {rank}")

    def cleanup(self):
        dist.destroy_process_group()
        print(f"Rank {rank} is done.")

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

    rank       = int(sys.argv[1])
    world_size = int(sys.argv[2])
    min_ddp = MinimalDDP()
    min_ddp.demo_basic(rank, world_size)
