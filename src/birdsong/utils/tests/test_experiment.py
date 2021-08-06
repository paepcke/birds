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

from birdsong.utils.experiment import Experiment


#*********TEST_ALL = True
TEST_ALL = False


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
        
    def tearDown(self):
        shutil.rmtree(self.exp_root)
        shutil.rmtree(self.prefab_exp_root)

# ------------------- Tests --------------

    #------------------------------------
    # test_creation
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_creation(self):
        exp = Experiment(self.exp_root)
        self.assertEqual(exp['root_path'], self.exp_root)
        self.assertEqual(exp['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp.csv_writers == {})
        
        # Should have a json file in root dir:
        self.assertTrue(os.path.exists(os.path.join(self.exp_root, 'experiment.json')))
        
        # Delete and restore the experiment:
        del exp
        exp1 = Experiment.load(self.exp_root)
        self.assertEqual(exp1['root_path'], self.exp_root)
        self.assertEqual(exp1['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp1.csv_writers == {})

    #------------------------------------
    # test_dict_addition 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dict_addition(self):
        
        exp = Experiment(self.exp_root)

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

        exp.close()
        
    #------------------------------------
    # test_saving
    #-------------------
    
    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving(self):
        exp = Experiment(self.exp_root)

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save(tst_dict, 'first_dict')
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp = Experiment.load(self.exp_root)
        
        # First, ensure that the test dict 
        # is unharmed without using the Experiment
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