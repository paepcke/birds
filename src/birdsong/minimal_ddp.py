#!/usr/bin/env python

import argparse
import copy
import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


class MinimalDDP:
    '''Test whether DDP really does something'''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, rank, test_goal):
        '''If set to 'parameters', save models
           after each iteration, and after the
           run, check how parms change.
           
           If set to 'drift', run many iterations,
           and observe two instances drift apart.
           
           Should differentiate between rank and
           local_rank. In this case they happen to
           be the same values.
        '''

        print(f"Test goal: {test_goal}")
        if test_goal == 'drift':
            self.epochs  = 5
            self.samples = 50
        else:
            self.epochs  = 2
            self.samples = 3

        self.test_goal = test_goal
        self.rank = rank

    #------------------------------------
    # setup
    #-------------------

    def setup(self, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=self.rank, world_size=world_size)

    #------------------------------------
    # demo_basic
    #-------------------

    def demo_basic(self, world_size, model_save_dir='/tmp'):
        '''The action: train model; save intermediate states'''
            
        print(f"Running basic DDP example on rank {self.rank}.")
        self.setup(world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(self.rank)
        ddp_model = DDP(model, device_ids=[self.rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # For saving model copies
        # before and after back prop
        # for each loop iteration:
        
        before = []
        after  = []
        
        for epoch in range(self.epochs):
            print(f"Rank{self.rank}: start epoch {epoch}")
            for _i in range(self.samples):
                
                optimizer.zero_grad()
                ddp_model(torch.randn(20, 10).to(self.rank))
                labels = torch.randn(20, 5).to(self.rank)
                
                # If checking parameter changes/sync:
                # Copy and save model copies before and
                # after back prop:
                if self.test_goal == 'parameters':
                    before.append(copy.deepcopy(ddp_model))
                    
                loss_fn(outputs, labels).backward()
                
                if self.test_goal == 'parameters':
                    after.append(copy.deepcopy(ddp_model))

                optimizer.step()

                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        if self.test_goal == 'parameters':
            dist.barrier()
            # Save the state_dirs of all before-prop
            # and after-prop model copies; each in its
            # own file:
            self.save_model_arrs(self.rank, before, after, model_save_dir)
        
        self.cleanup()

        if self.test_goal == 'parameters' and self.rank == 0:
                    # Using the saved files, 
            # verify that model parameters
            # change, and are synchronized
            # as expected:
            
            self.report_model_diffs()

    #------------------------------------
    # save_model_arrs 
    #-------------------
    
    def save_model_arrs(self, before_arr, after_arr, model_save_dir):
        '''Save state_dict of modesl in arrays to files'''
        
        print(f"Proc{self.rank}: saving arrays of before and after models.")
        
        for i, (model_before, model_after) in enumerate(zip(before_arr, after_arr)):
            model_before.cpu()
            model_after.cpu()
            torch.save(model_before.state_dict(),
                       os.path.join(model_save_dir, f"before_models_r{self.rank}_{i}.pth"))
            torch.save(model_after.state_dict(),
                       os.path.join(model_save_dir, f"after_models_r{self.rank}_{i}.pth"))

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
        print(f"Rank {self.rank} is done.")
        
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

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Test model parameter values or process drift"
                                     )

    parser.add_argument('-r', '--rank', type=int)
    parser.add_argument('-g', '--goal', choices=['parameters', 'drift'])
    args = parser.parse_args();

    test_goal      = args.goal
    world_size     = 2 
    model_save_dir = '/tmp'
    min_ddp = MinimalDDP(test_goal)
    # min_ddp = MinimalDDP('drift')
    min_ddp.demo_basic(args.rank, world_size, model_save_dir)
