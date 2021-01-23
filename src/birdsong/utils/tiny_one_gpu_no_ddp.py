'''
Created on Jan 22, 2021

@author: paepcke
'''

#!/usr/bin/env python

import os
import sys
import copy

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch import randn

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

class TinyNoDDP:
    '''Test whether DDP really does something'''
    
    epochs  = 2
    samples = 3

    #------------------------------------
    # demo_basic
    #-------------------

    def demo_basic(self):
        '''The action: train model; save intermediate states'''
            
        print(f"Running basic non-DDP example")
    
        model = ToyModel().to(0)
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # For saving model copies
        # before and after back prop
        # for each loop iteration:
        
        before = []
        after  = []
        
        for _epoch in range(self.epochs):
            for _i in range(self.samples):
                
                optimizer.zero_grad()
                outputs = model(randn(20, 10).to(0))
                labels = randn(20, 5).to(0)
                
                # Copy and save model copies before and
                # after back prop:
                before_model = model.cpu()
                before.append(copy.deepcopy(before_model.state_dict()))
                model.to(0)
                
                loss_fn(outputs, labels).backward()

                optimizer.step()
                
                after_model = model.cpu()
                after.append(copy.deepcopy(after_model.state_dict()))
                model.to(0)
                
                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()

        self.report_result(before, after)
        
        self.cleanup()

    #------------------------------------
    # report_result 
    #-------------------
    
    def report_result(self, before_arr, after_arr):
        '''Save state_dict of modesl in arrays to files'''
        
        # For each epoch
        for i in range(len(before_arr)):
            before_state = before_arr[i]
            after_state  = before_arr[i]
            
            # Start pessimistic:
            are_equal = True
            msg = f"Loop {i}: before/after states are "
            
            for before_tns, after_tns in zip(before_state.values(),
                                             after_state.values() 
                                             ):
                if before_tns.ne(after_tns).any():
                    are_equal = False
                
            if are_equal:
                msg += "equal"
            else:
                msg += "different"
            print(msg)

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

    min_ddp = TinyNoDDP()
    min_ddp.demo_basic()
