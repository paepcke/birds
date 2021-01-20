#!/usr/bin/env python

import os
import sys
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

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
                before.append(copy.deepcopy(ddp_model))
                loss_fn(outputs, labels).backward()
                after.append(copy.deepcopy(ddp_model))

                optimizer.step()

                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        dist.barrier()

        # Save the state_dirs of all before-prop
        # and after-prop model copies; each in its
        # own file:
        self.save_model_arrs(rank, before, after, model_save_dir)
        
        self.cleanup()
        
        if rank == 0:
            # Using the saved files, 
            # verify that model parameters
            # change, and are synchronized
            # as expected:
            
            self.report_model_diffs()

    #------------------------------------
    # save_model_arrs 
    #-------------------
    
    def save_model_arrs(self, rank, before_arr, after_arr, model_save_dir):
        '''Save state_dict of modesl in arrays to files'''
        
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

    def report_model_diffs(self, model_save_dir='/tmp'):
        '''Check that model parms changed or 
            were synched as expected '''
        
        model_arrs_len = self.epochs * self.samples
        
        # Among GPUs, model parms should differ
        # before backprop... 
        befores_differ_among_GPUs   = True    # that's the hope
        # ... but be synched by DDP after
        afters_differ_among_GPUs    = False   # that's the hope
        
        # Wihin a single GPU, the model should be 
        # changed by the backprop:
        befores_differ_from_afters  = True    # that's the hope
        
        for i in range(model_arrs_len):
            before_path_r0 = os.path.join(model_save_dir, f"before_models_r0_{i}.pth")
            before_path_r1 = os.path.join(model_save_dir, f"before_models_r1_{i}.pth")
            
            after_path_r0 = os.path.join(model_save_dir, f"after_models_r0_{i}.pth")
            after_path_r1 = os.path.join(model_save_dir, f"after_models_r1_{i}.pth")
            
            before_state0 = torch.load(before_path_r0)
            before_state1 = torch.load(before_path_r1)
            
            after_state0 = torch.load(after_path_r0)
            after_state1 = torch.load(after_path_r1)
            
            # The between-GPUs test:
            for (param_tns0, param_tns1) in zip(before_state0, before_state1):
                if before_state0[param_tns0].eq(before_state1[param_tns1]).all():
                    # Dang!
                    befores_differ_among_GPUs = False
            
            for (param_tns0, param_tns1) in zip(after_state0, after_state1):
                if after_state0[param_tns0].ne(after_state1[param_tns1]).any():
                    # Dang!
                    afters_differ_among_GPUs = False
                    
            # The within-GPUs test:
            for (param_tns_pre, param_tns_post) in zip(before_state0, after_state0):
                if before_state0[param_tns_pre].eq(before_state0[param_tns_post]).all():
                    # Dang!
                    befores_differ_from_afters = False
            
        if befores_differ_among_GPUs:
            print("Good: corresponding pre-backward model parms differ")
        else:
            print("Suspicious: corresponding pre-backward model parms match exactly")
            
        if afters_differ_among_GPUs:
            print("Bad: backward does not seem to broadcast parms")
        else:
            print("Good: corresponding post-backward model parms match exactly")

        # Within one GPU, model parms before and 
        # after back prop should be different.
        if befores_differ_from_afters:
            print("Good: back prop does change model parms")
        else:
            print("Suspicious: back prop has no impact on model parms") 


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

    rank           = int(sys.argv[1])
    world_size     = 2 # int(sys.argv[2])
    model_save_dir = '/tmp'
    min_ddp = MinimalDDP()
    min_ddp.demo_basic(rank, world_size, model_save_dir)
