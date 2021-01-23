#!/usr/bin/env python

import os
import sys
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import randn

from torch.nn.parallel import DistributedDataParallel as DDP

class MinimalDDP:
    '''Test whether DDP really does something'''
    
    epochs  = 2
    batches = 3

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

    def demo_basic(self, rank, world_size):
            
        print(f"Running basic DDP on two GPUs same machine: rank {rank}.")
        self.setup(rank, world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        dist.barrier()
        
        for epoch_num in range(self.epochs):
            for batch_num in range(self.batches):
                
                optimizer.zero_grad()
                outputs = ddp_model(randn(20, 10).to(rank))
                labels = randn(20, 5).to(rank)
                
                #********* Begin Portion of Interest ******
                before_model = ddp_model.cpu()
                before_state = copy.deepcopy(before_model.state_dict())
                if rank == 1:
                    torch.save(before_state, f"/tmp/before_rank1.pth")
                ddp_model.to(rank)
                
                loss_fn(outputs, labels).backward()
                optimizer.step()

                after_model = ddp_model.cpu()
                after_state = after_model.state_dict()
                if rank == 1:
                    torch.save(after_state, f"/tmp/after_rank1.pth")
                ddp_model.to(rank)
                                
                dist.barrier()
                
                # Read the other's before/after states:
                if rank == 0:
                    other_before_state = torch.load(f"/tmp/before_rank1.pth")
                    other_after_state  = torch.load(f"/tmp/after_rank1.pth")                
                
                    # Before states should be different:
                    states_equal = True
                    for before_parm, other_before_parm in zip(other_before_state.values(),
                                                              before_state.values()):
                        if before_parm.ne(other_before_parm).any():
                            states_equal = False
    
                    print(f"Epoch{epoch_num} batch{batch_num}: Before states across gpus are {('equal' if states_equal else 'different')}")


                    # After states should be the same:
                    states_equal = True
                    for after_parm_other, after_parm in zip(other_after_state.values(),
                                                       after_state.values()):
                        if after_parm_other.ne(after_parm).any():
                            states_equal = False
    
                    print(f"Epoch{epoch_num} batch{batch_num}: After states across gpus are {('equal' if states_equal else 'different')}")

                #********* End Portion of Interest ******
                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        dist.barrier()

        self.cleanup()

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

    # Started via minimal_ddp_launcher.py,
    # which sets the rank:
    
    rank           = int(sys.argv[1])
    world_size     = 2
    min_ddp = MinimalDDP().demo_basic(rank, world_size)
