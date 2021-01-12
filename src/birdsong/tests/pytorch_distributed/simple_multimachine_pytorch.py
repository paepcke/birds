'''
Created on Jan 12, 2021

@author: paepcke
'''

import os
import sys
import time

import torch
from torch.multiprocessing import Process

import torch.distributed as dist


class PytorchDistribTester:
    
    def __init__(self):

        self.master_addr = sys.argv[1]
        self.rank        = int(sys.argv[2])
        self.local_rank  = int(sys.argv[3])
        self.world_size  = int(sys.argv[4])
        self.master_port = 5678
        
        if   dist.is_nccl_available():
            self.backend = 'nccl'
        elif dist.is_gloo_available():
            self.backend = 'gloo'
            
        print(f"Using backend: {self.backend}")
    
    def fake_train_script(self, rank, world_size):
        """ Distributed function to be implemented later. """
        while True:
            print(f"Training (Rank: {rank}; world_size: {world_size})")
            time.sleep(2)

    def run(self):
        
        processes = []
        for rank in range(self.world_size):
            p = Process(target=self.init_process)
            p.start()
            processes.append(p)
    
        for p in processes:
            p.join()    

    def init_process(self, backend='nccl'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = str(self.master_addr)
        os.environ['MASTER_PORT'] = str(self.master_port)
        dist.init_process_group(self.backend, 
                                rank=self.rank, 
                                world_size=self.world_size)
        self.fake_train_script(self.rank, self.world_size)
    
    
if __name__ == '__main__':
    tester = PytorchDistribTester()
    tester.run()
