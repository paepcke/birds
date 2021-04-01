'''
Created on Jan 22, 2021

@author: paepcke
'''

#!/usr/bin/env python

import os
import sys
import copy

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
    batches = 3

    #------------------------------------
    # demo_basic
    #-------------------

    def demo_basic(self):
        '''The action: train model; save intermediate states'''
            
        print(f"Running basic non-DDP example")
    
        model = ToyModel().to(0)
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        for _epoch in range(self.epochs):
            for _i in range(self.batches):
                
                optimizer.zero_grad()
                # Batch size twenty, ten features:
                outputs = model(randn(20, 10).to(0))
                # Five classes: 
                labels = randn(20, 5).to(0)
                
                # Copy and save model copies before back prop:
                before_model = model.cpu()
                before_state = copy.deepcopy(before_model.state_dict())
                model.to(0)
                
                loss_fn(outputs, labels).backward()
                optimizer.step()
                
                after_model = model.cpu()
                after_state = after_model.state_dict()
                states_equal = True
                for before_parm, after_parm in zip(before_state.values(),
                                                   after_state.values()):
                    if before_parm.ne(after_parm).any():
                        states_equal = False
                model.to(0)
                print(f"Before and after are {('equal' if states_equal else 'different')}")
                
                # Clean GPU memory:
                outputs.cpu()
                labels.cpu()
        
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

    TinyNoDDP().demo_basic()
