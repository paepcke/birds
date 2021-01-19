'''
Created on Jan 19, 2021

@author: paepcke
'''
import os
import sys
import tempfile
import subprocess

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

class MinimalDDPLauncher:
   
    def run_demo(self, demo_fn, world_size):
        # mp.spawn(demo_fn,
        #          args=(world_size,),
        #          nprocs=world_size,
        #          join=True)
        procs = []
        for i in range(world_size):
            print(f"Starting {demo_fn}[{i}] of {world_size}")
            procs.append(subprocess.Popen([demo_fn, i, world_size]))

# ------------------------ Main ------------
if __name__ == '__main__':

    curr_dir = os.path.dirname(__file__)
    script_path = os.path.join(curr_dir, 'minimal_ddp.py')
    launcher = MinimalDDPLauncher()
    launcher.run_demo(script_path, 2)
