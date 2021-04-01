'''
Created on Jan 12, 2021

@author: paepcke
'''

import os
from subprocess import Popen
import subprocess
import sys

#*****from torch.multiprocessing import Process

import torch.distributed as dist


class PytorchDistribTester:
    
    def __init__(self):

        self.master_addr = sys.argv[1]
        self.rank        = int(sys.argv[2])
        self.local_rank  = int(sys.argv[3])
        self.world_size  = int(sys.argv[4])
        self.master_port = 5678
   
    def run(self):
        
        curr_dir = os.path.dirname(__file__)
        script_path = os.path.join(curr_dir, 'fake_train_script.py')
        p = Popen([script_path,
                   self.master_addr,
                   self.rank,
                   self.local_rank,
                   self.world_size
                   ], 
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE
                   )
        p.wait()
        

#     def init_process(self, rank):
#         """ Initialize the distributed environment. """
#         os.environ['MASTER_ADDR'] = str(self.master_addr)
#         os.environ['MASTER_PORT'] = str(self.master_port)
#         dist.init_process_group(self.backend, 
#                                 rank=self.rank, 
#                                 world_size=self.world_size)
#         self.fake_train_script(self.rank, self.world_size)
    
    
if __name__ == '__main__':
    tester = PytorchDistribTester()
    tester.run()
