'''
Created on Jan 19, 2021

@author: paepcke
'''
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


class MinimalDDP:

    def setup(self, rank, world_size):
        if sys.platform == 'win32':
            # Distributed package only covers collective communications with Gloo
            # backend and FileStore on Windows platform. Set init_method parameter
            # in init_process_group to a local file.
            # Example init_method="file:///f:/libtmp/some_file"
            init_method="file:///{your local file path}"
    
            # initialize the process group
            dist.init_process_group(
                "gloo",
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
        else:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
    
            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def demo_basic(self, rank, world_size):
        print(f"Running basic DDP example on rank {rank}.")
        self.setup(rank, world_size)
    
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])
    
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
    
        self.cleanup()

    def run_demo(self, demo_fn, world_size):
        mp.spawn(demo_fn,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    
    
    def cleanup(self):
        dist.destroy_process_group()


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

    min_ddp = MinimalDDP()
    min_ddp.run_demo(min_ddp.demo_basic, 1)