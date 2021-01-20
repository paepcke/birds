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

    def demo_basic(self, rank, world_size, model_save_dir):
            
        print(f"Running basic DDP example on rank {rank}.")
        self.setup(rank, world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        before = []
        after  = []
        for _epoch in range(self.epochs):
            for _i in range(self.samples):
                
                optimizer.zero_grad()
                outputs = ddp_model(torch.randn(20, 10))
                labels = torch.randn(20, 5).to(rank)
                
                before.append(copy.deepcopy(model))
                loss_fn(outputs, labels).backward()
                after.append(copy.deepcopy(model))

                optimizer.step()
                             
        dist.barrier()

        self.save_model_arrs(rank, before, after, model_save_dir)
        self.cleanup()
        
        if rank == 0:
            self.report_model_diffs(model_save_dir)

    #------------------------------------
    # save_model_arrs 
    #-------------------
    
    def save_model_arrs(self, rank, before_arr, after_arr, model_save_dir):
        
        print(f"Proc{rank}: saving arrays of before and after models.")
        
        for i, (model_before, model_after) in enumerate(zip(before_arr, after_arr)):
            model_before.cpu()
            model_after.cpu()
            torch.save(model_before.state_dict(),
                       os.path.join(model_save_dir, f"before_models_r{rank}_{i}.pth"))
            torch.save(model_after.state_dict(),
                       os.path.join(model_save_dir, f"after_models_r{rank}_{i}.pth"))

    #------------------------------------
    # report_model_diffs 
    #-------------------

    def report_model_diffs(self, model_save_dir):
        
        model_arrs_len = self.epochs * self.samples
        befores_differ = True
        afters_differ  = False
        
        for i in range(model_arrs_len):
            before_path_r0 = os.path.join(model_save_dir, f"before_models_r0_{i}.pth")
            before_path_r1 = os.path.join(model_save_dir, f"before_models_r1_{i}.pth")
            
            after_path_r0 = os.path.join(model_save_dir, f"after_models_r0_{i}.pth")
            after_path_r1 = os.path.join(model_save_dir, f"after_models_r1_{i}.pth")
            
            before_state0 = torch.load(before_path_r0)
            before_state1 = torch.load(before_path_r1)
            
            after_state0 = torch.load(after_path_r0)
            after_state1 = torch.load(after_path_r1)
            
            for (param_tns0, param_tns1) in zip(before_state0, before_state1):
                if before_state0[param_tns0].eq(before_state1[param_tns1]):
                    befores_differ = False
            
            for (param_tns0, param_tns1) in zip(after_state0, after_state1):
                if after_state0[param_tns0].ne(after_state1[param_tns1]):
                    afters_differ = False

        if befores_differ:
            print("Good: corresponding pre-backward model parms differ")
        else:
            print("Suspicious: corresponding pre-backward model parms match exactly")
            
        if afters_differ:
            print("Bad: backward does not seem to broadcast parms")
        else:
            print("Good: corresponding post-backward model parms match exactly")



    #------------------------------------
    # compare_model_parameters 
    #-------------------

    def compare_model_parameters(self, model, other):
        for parms1, parms_other in zip(model.parameters(), other.parameters()):
            if parms1.data.ne(parms_other.data).sum() > 0:
                return False
        return True        
        
        
        
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

    rank           = int(sys.argv[1])
    world_size     = 2 # int(sys.argv[2])
    model_save_dir = f"/home/paepcke/tmp/PytorchComm/"
    min_ddp = MinimalDDP()
    min_ddp.demo_basic(rank, world_size, model_save_dir)
