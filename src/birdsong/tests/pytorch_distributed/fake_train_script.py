'''
Created on Jan 12, 2021

@author: paepcke
'''

import os
import sys
import time

import torch.distributed as dist

class FakeTrainScript(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.master_addr = sys.argv[1]
        self.rank        = sys.argv[2]
        self.local_rank  = sys.argv[3]
        self.world_size  = sys.argv[4]
        
        os.environ['MASTER_ADDR'] = str(self.master_addr)
        os.environ['MASTER_PORT'] = '5678'
        os.environ['RANK']        = str(self.rank)
        os.environ['LOCAL_RANK']  = str(self.local_rank)
        os.environ['WORLD_SIZE']  = str(self.world_size)
        
        if   dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_gloo_available():
            backend = 'gloo'
            
        print(f"Using backend: {backend}")
        
        
        dist.init_process_group(backend=backend,
                                init_method='env://'
                                )

        print("Returned from process group init!!!!!")
        self.fake_train_script(self.rank, self.world_size)


    def fake_train_script(self, rank, world_size):
        """ Distributed function to be implemented later. """
        while True:
            print(f"Training (Rank: {rank}; world_size: {world_size})")
            time.sleep(2)
        
# ------------------------ Main ------------
if __name__ == '__main__':
    FakeTrainScript()