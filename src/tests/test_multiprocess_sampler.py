'''
Created on Dec 17, 2020

@author: paepcke
'''
'''
Created on Dec 17, 2020

@author: paepcke
'''
import os
import unittest

import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#*****from birds_train_parallel import PYTORCH_COMM_PORT
PYTORCH_COMM_PORT = 29920
from bird_dataset import BirdDataset
from samplers import DistributedSKFSampler

#******TEST_ALL = True
TEST_ALL = False

class TestMultiProcessSampler(unittest.TestCase):
    
    CURR_DIR = os.path.dirname(__file__)
    TEST_FILE_PATH_BIRDS = os.path.join(CURR_DIR, 'data/train')

    #------------------------------------
    # setUP 
    #-------------------

    def setUp(self):
        self.dataset = BirdDataset(self.TEST_FILE_PATH_BIRDS)

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        self.uninit_multiprocessing()

    #------------------------------------
    # test_dist_package_related 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dist_package_related(self):
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f"{PYTORCH_COMM_PORT}"
        
        if dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")
        
        # One machine, one GPU:
        world_size = 1
        rank       = 0
        dist.init_process_group(backend,
                                init_method=f'env://?world_size={world_size}&rank={rank}'
                                ) 
        self.assertTrue(dist.is_available())
        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_rank(), rank)
        self.assertEqual(dist.get_world_size(), world_size)
        
        dist.destroy_process_group()
        self.assertFalse(dist.is_initialized())


    #------------------------------------
    # test_one_machine_one_gpu
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_one_machine_one_gpu(self):
        
        try:
            DistributedSKFSampler(self.dataset)
            self.fail("Expected runtime error warning that dist.init_process_group() not called.")
        except RuntimeError:
            pass

        rank = 0
        world_size = 1
        self.init_multiprocessing(rank, world_size)

        sampler = DistributedSKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=False
            )
        
        # No folds served yet:
        self.assertEqual(sampler.folds_served, 0)
        
        # Since only one machine+GPU, this sampler
        # should have access to the whole dataset:
        self.assertEqual(len(sampler.my_indices), len(self.dataset))
        
        # Same number of samples must be refleced
        # in the sampler's length method: 

        self.assertEqual(len(sampler), 12)


    #------------------------------------
    # test_one_machine_two_gpus
    #-------------------

    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
#     def test_one_machine_two_gpu(self):
#         
#         rank = 0
#         world_size = 2
#         self.init_multiprocessing(rank, world_size)
#         
#         sampler = DistributedSKFSampler(
#             self.dataset,
#             num_folds=3,
#             shuffle=False
#             )
#         
#         # Two GPUs: this sampler should
#         # get 1/2 the dataset: 
#         self.assertEqual(len(sampler.my_indices), len(self.dataset) / 2)
#         
#         # Same number of samples must be refleced
#         # in the sampler's length method: 
# 
#         self.assertEqual(len(sampler), 12)


# ------------------- Utils ---------------
    #------------------------------------
    # init_multiprocessing
    #-------------------
    
    def init_multiprocessing(self, rank, world_size):
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f"{PYTORCH_COMM_PORT}"
        
        if dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")

        dist.init_process_group(backend,
                                init_method=f'env://?world_size={world_size}&rank={rank}'
                                ) 


    #------------------------------------
    # uninit_multiprocessing 
    #-------------------
    
    def uninit_multiprocessing(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
