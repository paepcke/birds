'''
Created on Dec 19, 2020

@author: paepcke
'''
import os, sys
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

import unittest

from birdsong.utils.dottable_map import DottableMap 

TEST_ALL = True
#TEST_ALL = False

class TestDottableMap(unittest.TestCase):


    def setUp(self):
        self.m = DottableMap({'first_name': 'Eduardo'}, # Will flatten
                             last_name='Pool', 
                             age=24, 
                             sports=['Soccer'],
                             pi=3.14159,
                             nested=DottableMap({'profession' : 'engineer',
                                                'company' : 'HP'
                                                }),
                             dbl_nest = DottableMap({'profession' : 'engineer',
                                                'company' : DottableMap({'name' : 'HP',
                                                                         'location' : 'Palo Alto'
                                                                         })
                                                }), 
                             )



    def tearDown(self):
        pass


    #------------------------------------
    # test_creation 
    #-------------------

    def test_creation(self):
        # Dict argument is flattened into this DottableMap:
        self.assertEqual(self.m.first_name, 'Eduardo')

        # Keyword arg:
        self.assertEqual(self.m.age, 24)
        
        # Preserve lists as values:
        self.assertEqual(self.m.sports, ['Soccer'])
        
        # Nested DottableMap: retrieving the nested
        # DottableMap as a dict:
        self.assertDictEqual(self.m.nested, {'profession': 'engineer', 'company': 'HP'})
        
        # Nested DottableMap: retrieving elements from
        # the nested DottableMap:
        self.assertEqual(self.m.nested.company, 'HP')
        
        # Element from nested in nested:
        self.assertEqual(self.m.dbl_nest.company.location, 'Palo Alto')

    #------------------------------------
    # test_get_typed 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_get_typed(self):
        self.assertEqual(self.m.getint('age'), 24)
        self.assertEqual(self.m.getfloat('pi'), 3.14159)

    #------------------------------------
    # test_defaulting 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_defaulting(self):
        self.assertEqual(self.m.getint('foo', 10), 10)
        self.assertEqual(self.m.getfloat('bar', 2.5), 2.5)

# -------------- Main --------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
