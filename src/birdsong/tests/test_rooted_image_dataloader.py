'''
Created on Dec 13, 2020

@author: paepcke
'''
from _collections import OrderedDict
import os
import unittest

import natsort

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.rooted_image_dataset import MultiRootImageDataset


#*****TEST_ALL = True
TEST_ALL = False

#**************
import socket, sys
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
#***********

class TestNRootsImageDataset(unittest.TestCase):

    CURR_DIR = os.path.dirname(__file__)
    
    TEST_FILE_PATH_BIRDS = os.path.join(CURR_DIR, 'data/birds')
    TEST_FILE_PATH_CARS  = os.path.join(CURR_DIR, 'data/cars')
    TEST_FILE_PATH_GLOB  = os.path.join(CURR_DIR, 'data')

    #------------------------------------
    # setUpClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        cls.sample_class_assignments = \
            OrderedDict([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), 
                         (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11,1)])

    
    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        pass

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        pass

# ----------------- Testing SingleRootImageDataset Class -------

    #------------------------------------
    # testStructuresCreation 
    #-------------------

    #*******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testStructuresCreation(self):
        
        #************
        self.TEST_FILE_PATH_BIRDS = '/home/data/birds/recombined_data'
        #************
        ds = SingleRootImageDataset(self.TEST_FILE_PATH_BIRDS)
        
        # List of (class_id, class_name) 2-tples:
        maybe_dirty_dir_set = set(os.listdir(self.TEST_FILE_PATH_BIRDS))
        
        # Dataset skips dirs that start with a dot:
        dir_set = set([dir_name 
                         for dir_name 
                          in maybe_dirty_dir_set
                          if not dir_name.startswith('.')
                          ])
        
        class_id_assignments = enumerate(natsort.natsorted(dir_set))
        
        for class_id, class_name in class_id_assignments:
            self.assertEqual(ds.class_to_id[class_name], class_id)

        
        self.assertEqual(ds.sample_id_to_class,
                         self.sample_class_assignments)
        
        self.assertEqual(len(ds.sample_id_to_path), 
                         len(ds.sample_id_to_class))
        
    #------------------------------------
    # testGetItem 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testGetItem(self):
        ds = SingleRootImageDataset(self.TEST_FILE_PATH_BIRDS)
        (_img, class_id) = ds[0]
        self.assertEqual(class_id, 0)

# ----------------- Testing MultirootImageDataset Class -------

    #------------------------------------
    # test_multiroot_one_root 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multiroot_one_root(self):
        
        # Single root, degenerate case equivalent
        # to SingleRootImageDataset:
        
        mrds = MultiRootImageDataset(self.TEST_FILE_PATH_CARS)
        
        self.assertDictEqual(mrds.class_to_id,
                             {'audi' : 0,
                              'bmw'  : 1})
        self.assertDictEqual(mrds.sample_id_to_class,
                                OrderedDict({
                                    0 : 0,
                                    1 : 0,
                                    2 : 0,
                                    3 : 0,
                                    4 : 0,
                                    5 : 0,
                                    6 : 1,
                                    7 : 1,
                                    8 : 1,
                                    9 : 1,
                                   10 : 1,
                                   11 : 1,
                                    }))
        self.assertListEqual(mrds.class_id_list(), [0,1])
        self.assertDictEqual(mrds.sample_id_to_path,
                             OrderedDict({
                                    0 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi1.jpg'),
                                    1 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi2.jpg'),
                                    2 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi3.jpg'),
                                    3 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi4.jpg'),
                                    4 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi5.jpg'),
                                    5 : os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi6.jpg'),
                                    6 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw1.jpg'),
                                    7 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw2.jpg'),
                                    8 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw3.jpg'),
                                    9 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw4.jpg'),
                                   10 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw5.jpg'),
                                   11 : os.path.join(self.TEST_FILE_PATH_CARS, 'bmw', 'bmw6.jpg') 
                                   })
                             )
        
    #------------------------------------
    # test_multiroot_two_roots 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multiroot_two_roots(self):
        
        # Single root, degenerate case equivalent
        # to SingleRootImageDataset:
        
        mrds = MultiRootImageDataset([self.TEST_FILE_PATH_CARS,
                                      self.TEST_FILE_PATH_BIRDS])
        
        self.assertDictEqual(mrds.class_to_id,
                             {'audi'     : 0,
                              'bmw'      : 1,
                              'DYSMEN_S' : 2,
                              'HENLES_S' : 3
                              })

        # Total number of samples:
        #   6 Audis
        #   6 BMWs
        #   6 DYSMEN_S
        #   6 HENLES_S
        
        # Lenghts of dataset bookkeeping structs should
        # be num of samples:
        self.assertEqual(len(mrds.sample_id_to_path), 24)
        self.assertEqual(len(mrds.sample_id_to_class), 24)
        
        # Sample IDs should range from 0 to 23:
        self.assertEqual(min(mrds.sample_id_to_path.keys()), 0)
        self.assertEqual(max(mrds.sample_id_to_path.keys()), 23)

        self.assertEqual(min(mrds.sample_id_to_class.keys()), 0)
        self.assertEqual(max(mrds.sample_id_to_class.keys()), 23)

        # Path spot checks:
        
        # First path (for sample_id 0 should be the first audi:
        self.assertEqual(mrds.sample_id_to_path[0],
                         os.path.join(self.TEST_FILE_PATH_CARS, 'audi', 'audi1.jpg'))
        
        # Test path correctness for last BMW:
        bmw_paths    = natsort.natsorted(os.listdir(os.path.join(self.TEST_FILE_PATH_CARS, 
                                                              'bmw')))
        last_bmw_path = bmw_paths[-1]
        last_bmw_name = os.path.basename(last_bmw_path)
         
        bmw_class_id = mrds.class_to_id['bmw']
         
        bmw_sample_ids = [sample_id 
                          for sample_id 
                          in mrds.sample_id_to_class
                          if mrds.sample_id_to_class[sample_id] == bmw_class_id
                          ]
        last_bmw_sample_id = bmw_sample_ids[-1]
        self.assertTrue(os.path.basename(mrds.sample_id_to_path[last_bmw_sample_id]),
                        last_bmw_name)

    #------------------------------------
    # test_glob 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_glob(self):
        
        mrds = MultiRootImageDataset(self.TEST_FILE_PATH_GLOB)
        self.assertDictEqual(mrds.class_to_id,
                             OrderedDict([('DYSMEN_S', 0), 
                                          ('HENLES_S', 1), 
                                          ('audi', 2), 
                                          ('bmw', 3), 
                                          ('diving_gear', 4), 
                                          ('office_supplies', 5)])
                             )

        self.assertEqual(mrds.class_names(),
                         ['DYSMEN_S', 'HENLES_S', 'audi', 'bmw', 'diving_gear', 'office_supplies']
                         )

        # Spot check the 'snorkel.jpg' image file 
        # class assignment:
        
        # Function to return true if value in a given
        # (key,value) pair is a path ending in 'snorkel.jpg':
        snorkel_finder = lambda id_path_pair: os.path.basename(id_path_pair[1]) == 'snorkel.jpg'
        
        # Get sample_id for the snorkel.jpg image:
        snorkel_sample_ids = self.key_from_value(mrds.sample_id_to_path, 
                                                 fn=snorkel_finder)
        snorkel_sample_id  = snorkel_sample_ids[0]
        
        # Get the class ID from snorkel's sample_id:
        snorkel_class_id = mrds.sample_id_to_class[snorkel_sample_id]

        # Find the class name whose class ID is that
        # of the class to which snorkel.jpg maps:
        class_name_for_snorkel = self.key_from_value(mrds.class_to_id, 
                                                     sought_value=snorkel_class_id)[0]
                                                     
        # Should be 'diving_gear':
        self.assertEqual(class_name_for_snorkel, 'diving_gear')

    #------------------------------------
    # test_key_from_value 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_key_from_value(self):
        d = {'foo' : 10, 'bar' : 20}
        self.assertEqual(self.key_from_value(d, sought_value=20), ['bar'])
        
        fn = lambda key_val_pair: key_val_pair[1] > 10
        self.assertEqual(self.key_from_value(d, fn=fn), ['bar'])
        
        # Multiple keys map to the same value:
        # expect list of all the matching keys:
        d1 = {'foo' : 10, 'bar' : 20, 'fum' : 10}
        self.assertEqual(self.key_from_value(d1, sought_value=10),
                         ['foo', 'fum']
                         )

# ---------------- Utilities -----------------

    def key_from_value(self, the_dict, sought_value=None, fn=None):
        '''
        Given a dict, returns all of the dict's keys
        that for which the value matches a given condition.
        The condition can either be a value, or a function
        that given a (key,value) tuple returns True if value
        meets the desired condition.
        
        NOTE: should only be used for testing. If reverse
              lookup is really needed, create a reverse dict
              instead.
         
        @param the_dict: dict to search
        @type the_dict: dictionary
        @param sought_value: constant that is sought among the dict's values
        @type sought_value: any
        @param fn: function(key_val_tuple) --> True/False
        @type fn: function
        @returns list of keys for which dict[key] matches
            the condition
        @rtype hashable
        '''
        
        if sought_value is None and fn is None:
            raise ValueError("Arguments sought_value and fn must not both be None")

        if sought_value is not None and fn is not None:
            raise ValueError("One of sought_value and fn must be provided")
        
        if fn is None:
            value_filter = filter(lambda key_val_pair: key_val_pair[1] == sought_value,
                                  the_dict.items())
        else:
            value_filter = filter(fn, the_dict.items())
            
        return [(k,v)[0] for k,v in value_filter]

# ------------------- Main ----------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
