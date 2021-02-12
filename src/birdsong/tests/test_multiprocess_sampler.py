'''
Created on Dec 17, 2020

@author: paepcke
'''

import os
import unittest

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.samplers import DistributedSKFSampler
from birdsong.utils.dottable_config import DottableConfigParser
import torch.distributed as dist


#*****from birds_train_parallel import PYTORCH_COMM_PORT
PYTORCH_COMM_PORT = 29920

TEST_ALL = True
#TEST_ALL = False

class TestMultiProcessSampler(unittest.TestCase):
    
    CURR_DIR = os.path.dirname(__file__)

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        cls.config = DottableConfigParser(os.path.join(cls.CURR_DIR,
                                                       'bird_trainer_tst.cfg'
                                                       ))
        parent = os.path.join(cls.CURR_DIR, '..')
        cls.data_path = cls.config.getpath('Paths', 
                                           'root_train_test_data', 
                                           relative_to=parent)

    #------------------------------------
    # setUP 
    #-------------------

    def setUp(self):
        self.dataset = SingleRootImageDataset(self.data_path)

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

        self.assertEqual(len(sampler), len(self.dataset))


    #------------------------------------
    # test_one_machine_two_gpus
    #-------------------

    # Can't simulate multiple machines, because
    # they hang, waiting for each other.
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
#     def test_one_machine_two_gpu(self):
#          
#         rank = 1
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
        try:
            dist.destroy_process_group()
        except RuntimeError:
            # No process group was initialized
            # before.
            pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
