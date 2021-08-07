'''
Created on Aug 5, 2021

@author: paepcke
'''
import csv
import os
from pathlib import Path
import shutil
import struct
import unittest
import zlib

import torch
import pandas as pd

from birdsong.utils.experiment_manager import ExperimentManager


#******TEST_ALL = True
TEST_ALL = False

'''
TODO:
   o test moving the experiment: ensure relative addressing!
   o deleting an item
'''

class TestExperiment(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.exp_fname = 'experiment'
        cls.prefab_exp_fname = 'fake_experiment'
        
        cls.exp_root = os.path.join(cls.curr_dir, cls.exp_fname)
        cls.prefab_exp_root = os.path.join(cls.curr_dir, cls.prefab_exp_fname)
    
    def setUp(self):
        
        try:
            shutil.rmtree(self.prefab_exp_root)
        except FileNotFoundError:
            pass

        os.makedirs(self.prefab_exp_root)
        
        # Create a little torch model and save it:
        models_dir = os.path.join(self.prefab_exp_root,'models')
        os.makedirs(models_dir)
        model_path = os.path.join(models_dir, 'tiny_model.pth')
        tiny_model = TinyModel()
        torch.save(tiny_model, model_path)
        
        # Create two little csv files:
        csvs_dir   = os.path.join(self.prefab_exp_root,'csv_files')
        os.makedirs(csvs_dir)
        self.make_csv_files(csvs_dir)
        
        # Create a tiny png file:
        figs_dir   = os.path.join(self.prefab_exp_root,'figs')
        os.makedirs(figs_dir)
        with open(os.path.join(figs_dir, "tiny_png.png") ,"wb") as fd:
            fd.write(self.makeGrayPNG([[0,255,0],[255,255,255],[0,255,0]]))
            
        hparams_dir = os.path.join(self.prefab_exp_root,'hparams')
        os.makedirs(hparams_dir)
        self.hparams_path = self.make_neural_net_config_file(hparams_dir)
        
    def tearDown(self):
        try:
            self.exp.close()
        except:
            pass
        shutil.rmtree(self.exp_root)
        shutil.rmtree(self.prefab_exp_root)

# ------------------- Tests --------------

    #------------------------------------
    # test_creation
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_creation(self):
        exp = ExperimentManager(self.exp_root)
        self.assertEqual(exp['root_path'], self.exp_root)
        self.assertEqual(exp['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp.csv_writers == {})
        
        # Should have a json file in root dir:
        self.assertTrue(os.path.exists(os.path.join(self.exp_root, 'experiment.json')))
        
        # Delete and restore the experiment:
        exp.close()
        del exp
        
        exp1 = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        self.assertEqual(exp1['root_path'], self.exp_root)
        self.assertEqual(exp1['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp1.csv_writers == {})

    #------------------------------------
    # test_dict_addition 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dict_addition(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save(tst_dict, 'first_dict')
        
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            self.assertEqual(reader.fieldnames, ['foo', 'bar'])
            row_dict = next(reader)
            self.assertEqual(list(row_dict.values()), ['10','20'])
            self.assertEqual(list(row_dict.keys()), ['foo', 'bar'])

        writers_dict = exp.csv_writers
        self.assertEqual(len(writers_dict), 1)
        
        wd_keys   = list(writers_dict.keys())
        first_key = wd_keys[0]
        self.assertEqual(first_key, Path(csv_file_path).stem)
        self.assertEqual(type(writers_dict[first_key]), csv.DictWriter)

        # Add second row to the same csv:
        row2_dict = {'foo' : 100, 'bar' : 200}
        exp.save(row2_dict, 'first_dict')
        
        # Second row should be [100, 200]:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            row_dict0 = next(reader)
            self.assertEqual(list(row_dict0.values()), ['10','20'])
            row_dict1 = next(reader)
            self.assertEqual(list(row_dict1.values()), ['100','200'])

        # Should be able to just write a row, not a dict:
        exp.save([1000,2000], 'first_dict')
        # Look at 3rd row should be ['1000', '2000']:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            row_dict0 = next(reader)
            self.assertEqual(list(row_dict0.values()), ['10','20'])
            row_dict1 = next(reader)
            self.assertEqual(list(row_dict1.values()), ['100','200'])
            row_dict2 = next(reader)
            self.assertEqual(list(row_dict2.values()), ['1000','2000'])
        
    #------------------------------------
    # test_saving_csv
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_csv(self):
        exp = ExperimentManager(self.exp_root)

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save(tst_dict, 'first_dict')
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        # First, ensure that the test dict 
        # is unharmed without using the ExperimentManager
        # instance:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            self.assertEqual(reader.fieldnames, ['foo', 'bar'])
            row_dict = next(reader)
            self.assertEqual(list(row_dict.values()), ['10','20'])
            self.assertEqual(list(row_dict.keys()), ['foo', 'bar'])

        # Now treat the experiment
        writers_dict = exp.csv_writers
        self.assertEqual(len(writers_dict), 1)

    #------------------------------------
    # test_saving_hparams
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_hparams(self):
        
        exp = ExperimentManager(self.exp_root)

        exp.add_hparams(self.hparams_path)
        config_obj = exp['config']
        
        # The config instance should be available
        # via the config key:
        self.assertEqual(config_obj, exp['config'])
        self.assertEqual(config_obj['Training']['net_name'], 'resnet18')
        self.assertEqual(config_obj.getint('Parallelism', 'master_port'), 5678)

        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp1 = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        
        config_obj = exp1['config']
        self.assertEqual(config_obj['Training']['net_name'], 'resnet18')
        self.assertEqual(config_obj.getint('Parallelism', 'master_port'), 5678)

    #------------------------------------
    # test_saving_dataframes
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_dataframes(self):
        
        exp = ExperimentManager(self.exp_root)

        df = pd.DataFrame([[1,2,3],
                           [4,5,6],
                           [7,8,9]], 
                           columns=['foo','bar','fum'], 
                           index= ['row1','row2','row3'])

        # Save without the row labels (i.e w/o the index):
        dst_without_idx = exp.save(df, 'mydf')
        self.assertEqual(dst_without_idx, exp.csv_writers['mydf'].fd.name)

        df_retrieved_no_idx_saved = pd.read_csv(dst_without_idx)
        # Should have:
        #        foo  bar  fum
        #    0    1    2    3
        #    1    4    5    6
        #    2    7    8    9
        
        df_true_no_idx = pd.DataFrame.from_dict({'foo' : [1,4,7], 'bar' : [2,5,8], 'fum' : [3,6,9]})
        self.assertTrue((df_retrieved_no_idx_saved == df_true_no_idx).all().all())

        # Now save with index:
        dst_with_idx = exp.save(df, 'mydf_with_idx', index_col='My Col Labels')
        df_true_with_idx = df_true_no_idx.copy()
        df_true_with_idx.index = ['row1', 'row2', 'row3']
        df_retrieved_with_idx_saved = pd.read_csv(dst_with_idx, index_col='My Col Labels')
        # Should have:
        #           foo  bar  fum
        #    row1    1    2    3
        #    row2    4    5    6
        #    row3    7    8    9
        self.assertTrue((df_retrieved_with_idx_saved == df_true_with_idx).all().all())
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp1 = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        
        # All the above tests should work again:
        
        df_no_idx   = pd.read_csv(exp1.abspath('mydf', 'csv'))
        df_with_idx = pd.read_csv(exp1.abspath('mydf_with_idx', 'csv'), 
                                  index_col='My Col Labels')
        self.assertTrue((df_no_idx == df_true_no_idx).all().all())
        self.assertTrue((df_with_idx == df_true_with_idx).all().all())

    #------------------------------------
    # test_saving_series
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_series(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        my_series = pd.Series([1,2,3], index=['One', 'Two', 'Three'])
        
        dst = exp.save(my_series, 'series_test')
        # Get a dataframe whose first row is the series:
        series_read = pd.read_csv(dst)
        first_row   = series_read.iloc[0,:] 
        self.assertTrue((first_row == my_series).all())

        exp.close()
        
    #------------------------------------
    # test_saving_mem_items
    #-------------------
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_mem_items(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        my_dict = {'foo' : 10, 'bar' : 20}
        exp['my_dict'] = my_dict
        self.assertDictEqual(exp['my_dict'], my_dict)
        
        exp.save()
        
        exp1 = ExperimentManager.load(self.exp_root)
        self.assertDictEqual(exp['my_dict'], my_dict)

        animal_dict = {'horse' : 'big', 'mouse' : 'small'}
        exp1['my_dict'] = animal_dict
        self.assertDictEqual(exp['my_dict'], my_dict)
        self.assertDictEqual(exp1['my_dict'], animal_dict)

        exp1.save()

        # Without loading, exp still has my_dict:
        self.assertDictEqual(exp['my_dict'], my_dict)
        
        # But after reloading exp, the value should change:
        exp = ExperimentManager.load(self.exp_root)
        self.assertDictEqual(exp['my_dict'], animal_dict)

    #------------------------------------
    # test_abspath
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_abspath(self):
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save(tst_dict, 'first_dict')

        self.assertEqual(exp.abspath('first_dict', 'csv'), csv_file_path)

# -------------------- Utilities --------------

    def make_csv_files(self, dst_dir):
        csv1 = os.path.join(dst_dir, 'tiny_csv1.csv')
        csv2 = os.path.join(dst_dir, 'tiny_csv2.csv')
        
        with open(csv1, 'w') as fd:
            writer = csv.DictWriter(fd, fieldnames=['foo', 'bar'])
            writer.writeheader()
            writer.writerow({'foo' : 10, 'bar' : 20})
            writer.writerow({'foo' : 100,'bar' : 200})
            
        with open(csv2, 'w') as fd:
            writer = csv.DictWriter(fd, fieldnames=['blue', 'green'])
            writer.writeheader()
            writer.writerow({'blue' : 'sky', 'green' : 'grass'})
            writer.writerow({'blue' : 'umbrella', 'green' : 'forest'})

    #------------------------------------
    # makeGrayPNG
    #-------------------

    def makeGrayPNG(self, data, height = None, width = None):
        '''
        From  https://stackoverflow.com/questions/8554282/creating-a-png-file-in-python
        
        Converts a list of list into gray-scale PNG image.
        __copyright__ = "Copyright (C) 2014 Guido Draheim"
        __licence__ = "Public Domain"
        
        Usage:
            with open("/tmp/tiny_png.png","wb") as f:
                f.write(makeGrayPNG([[0,255,0],[255,255,255],[0,255,0]]))
        
        :return a png structure
        '''
        
        def I1(value):
            return struct.pack("!B", value & (2**8-1))
        def I4(value):
            return struct.pack("!I", value & (2**32-1))
        # compute width&height from data if not explicit
        if height is None:
            height = len(data) # rows
        if width is None:
            width = 0
            for row in data:
                if width < len(row):
                    width = len(row)
        # generate these chunks depending on image type
        makeIHDR = True
        makeIDAT = True
        makeIEND = True
        png = b"\x89" + "PNG\r\n\x1A\n".encode('ascii')
        if makeIHDR:
            colortype = 0 # true gray image (no palette)
            bitdepth = 8 # with one byte per pixel (0..255)
            compression = 0 # zlib (no choice here)
            filtertype = 0 # adaptive (each scanline seperately)
            interlaced = 0 # no
            IHDR = I4(width) + I4(height) + I1(bitdepth)
            IHDR += I1(colortype) + I1(compression)
            IHDR += I1(filtertype) + I1(interlaced)
            block = "IHDR".encode('ascii') + IHDR
            png += I4(len(IHDR)) + block + I4(zlib.crc32(block))
        if makeIDAT:
            raw = b""
            for y in range(height):
                raw += b"\0" # no filter for this scanline
                for x in range(width):
                    c = b"\0" # default black pixel
                    if y < len(data) and x < len(data[y]):
                        c = I1(data[y][x])
                    raw += c
            compressor = zlib.compressobj()
            compressed = compressor.compress(raw)
            compressed += compressor.flush() #!!
            block = "IDAT".encode('ascii') + compressed
            png += I4(len(compressed)) + block + I4(zlib.crc32(block))
        if makeIEND:
            block = "IEND".encode('ascii')
            png += I4(0) + block + I4(zlib.crc32(block))
        return png

    #------------------------------------
    # make_neural_net_config_file
    #-------------------
    
    def make_neural_net_config_file(self, dst_dir):
        '''
        Create a fake neural net configurations file.
        
        :param dst_dir: where to put the 'config.cfg' file
        :type dst_dir: src
        :return full path to new config file
        :rtype src
        '''
        txt = '''
            [Paths]
            
            # Root of the data/test files:
            root_train_test_data = /foo/bar
            
            [Training]
            
            net_name      = resnet18
            # Some comment
            pretrained     = True
            freeze         = 0
            min_epochs    = 2
            max_epochs    = 2
            batch_size    = 2
            num_folds     = 2
            opt_name      = Adam
            loss_fn       = CrossEntropyLoss
            weighted      = True
            kernel_size   = 7
            lr            = 0.01
            momentum      = 0.9
            
            [Parallelism]
            
            independent_runs = True
            master_port = 5678
            
            [Testing]
            
            num_classes = 32
            '''
        hparams_path = os.path.join(dst_dir, 'config.cfg')
        with open(hparams_path, 'w') as fd:
            fd.write(txt)
            
        return hparams_path

# -------------------- TinyModel --------------
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64 * 64, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

# --------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()