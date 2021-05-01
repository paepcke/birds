'''
Created on Apr 25, 2021

@author: paepcke
'''
import os
import unittest

from data_augmentation.utils import Utils
from data_augmentation.utils import AugmentationGoals

TEST_ALL = True
#TEST_ALL = False



class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.spectros_dir = os.path.join(cls.cur_dir, 'spectro_data')

    def setUp(self):
        pass


    def tearDown(self):
        pass

# ----------------------- Tests ---------------


    #------------------------------------
    # test_orig_file_name 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_orig_file_name(self):
        
        # Identity:
        aug_nm = "foo.wav"
        orig = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, aug_nm)
        
        aug_nm = "Amaziliadecora1061880-volume-10.wav"
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061880.wav')
        
        aug_nm = 'Amaziliadecora1061883-rain_bgd0ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061883.wav')
        
        aug_nm = 'Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'Amaziliadecora1061886.wav')
        
        # With directory relative:
        aug_nm = 'foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, 'foo/bar/Amaziliadecora1061886.wav')
        
        # With directory absolute:
        aug_nm = '/foo/bar/Amaziliadecora1061886-shift4600ms.wav'
        orig   = Utils.orig_file_name(aug_nm)
        self.assertEquals(orig, '/foo/bar/Amaziliadecora1061886.wav')


    #------------------------------------
    # test_listdir_abs 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_listdir_abs(self):
        
        # Get the built-in directory listing
        # with just the file names:
        nearly_truth = os.listdir(self.cur_dir)
        
        abs_paths = Utils.listdir_abs(self.cur_dir)
        self.assertEquals(len(nearly_truth), len(abs_paths))
        
        # Check existence of first file or dir:
        self.assertTrue(os.path.exists(abs_paths[0]))

    #------------------------------------
    # test_sample_compositions_by_species 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_sample_compositions_by_species(self):
        
        dist_df = Utils.sample_compositions_by_species(self.spectros_dir)
        #truth =  pd.DataFrame.from_dict({'AMADEC' : 1, 'FORANA' : 5}, orient='index', columns=['num_samples'])
        self.assertListEqual(list(dist_df.columns), ['num_samples'])
        self.assertEqual(int(dist_df.loc['AMADEC']), 1)
        self.assertEqual(int(dist_df.loc['FORANA']), 5)
        
    #------------------------------------
    # test_compute_num_augs_per_species 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_num_augs_per_species(self):
        
        aug_volumes = AugmentationGoals.MAX
        sample_distrib_df = Utils.sample_compositions_by_species(self.spectros_dir)
        augs_to_do = Utils.compute_num_augs_per_species(aug_volumes, sample_distrib_df)
        self.assertEqual(augs_to_do['AMADEC'], 4)
        self.assertEqual(augs_to_do['FORANA'], 0)


    #------------------------------------
    # test_sample_compositions_by_species
    #-------------------
    
    #@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    #def test_sample_compositions_by_species(self):
    #    path = "../TAKAO_BIRD_WAV_feb20_augmented_samples-0.33n-0.33ts-0.33w-exc/spectrograms_augmented/"
    #    df = Utils.sample_compositions_by_species(path, True)
    #    print(df)
    
    #------------------------------------
    # test_rec_len_compositions_by_species 
    #-------------------

    #@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    #def test_rec_len_compositions_by_species(self):
    #    path = "../TAKAO_BIRD_WAV_feb20/"
    #    df = Utils.recording_lengths_by_species(path)
    #    print(df)
# ---------------- Main --------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()