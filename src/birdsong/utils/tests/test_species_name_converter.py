'''
Created on Jun 7, 2021

@author: paepcke
'''
import unittest

from birdsong.utils.species_name_converter import SpeciesNameConverter, \
    DIRECTION, ConversionError


TEST_ALL = True
#TEST_ALL = False


class SpeciesNameConverterTester(unittest.TestCase):

    all_xeno_canto = [
    	'AMADEC',  'ARRCON', 'CORALT_C', 'DYSMEN_S', 
        'FORANA',  'HYLDEC','LOPPIT','MYISIM','PHERUT',
        'RUPMAG','THREPI','ZIMPAR',	'ARRAUR_C','AUTEXS', 
        'CORALT_S', 'EMPFLT',   'HENLES_C','LEPSOU',
        'MELRUB','MYITUB','POLPLU','TANGYR','TODCIN',
    	'ARRAUR_S','BROJUG', 'DYSMEN_C', 'EUPIMI', 'HENLES_S',
        'LEPVEX','MILCHI','PHAGUY','RAMAMB','TANICT','TURASS'
        ]
    
    all_labeled = [
        'bana','bcmm','bgta','blgd','bssp','btsa',
        'cmto','ctfl','gaga','ghta','grki','howp',
        'howr','paty','rbps','rcwp','roha','rthu',
        'shwc','snhu','sofl','srta','stsa','vase',
        'wcpa','wtdo','yceu','yofl'
        ]
    
    @classmethod
    def setUpClass(cls):
        cls.cnv = SpeciesNameConverter()

    def setUp(self):
        pass


    def tearDown(self):
        pass


    #------------------------------------
    # test_from_four 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_four(self):
        
        # Test everying about the four-letter 
        # from-direction:
        self.assertEqual(self.cnv['WCPA', DIRECTION.FOUR_SCI], 'PIONUS SENILIS')
        self.assertEqual(self.cnv['wcpa', DIRECTION.FOUR_SCI], 'PIONUS SENILIS')
        self.assertEqual(self.cnv['wcpa', DIRECTION.FOUR_SIX], 'PIOSEN')

        # Test keys()        
        keys = list(self.cnv.keys(DIRECTION.FOUR_SIX))
        self.assertEqual(len(keys[0]), 4)

        # Test values()
        vals = list(self.cnv.values(DIRECTION.FOUR_SIX))
        self.assertEqual(len(vals[0]), 6)
        vals = list(self.cnv.values(DIRECTION.FOUR_SCI))
        self.assertGreater(len(vals[0]), 6)

        # Test items()
        for key, val in self.cnv.items(DIRECTION.FOUR_SIX):
            self.assertEqual(len(key), 4)
            self.assertEqual(len(val), 6)
            break
            
        for key, val in self.cnv.items(DIRECTION.FOUR_SCI):
            self.assertEqual(len(key), 4)
            self.assertGreater(len(val), 6)
            break

        # Test four to five for species without song/call split:
        for five_code in self.cnv.ALL_OCCURRING_SPECIES:
            four_code = self.cnv[five_code, DIRECTION.FIVE_FOUR]
            self.assertEqual(four_code, five_code[:4])
            
        # Going four to five should yield 4-code with 'G' appended,
        # BUT: all species that must be split into call/song should
        # generate a ConversionError:

        for five_code in self.cnv.ALL_OCCURRING_SPECIES:
            # All should work, as tested above...
            four_code = self.cnv[five_code, DIRECTION.FIVE_FOUR]
            self.assertEqual(len(four_code), 4)
            # but:
            try:
                looked_up_five_code = self.cnv[four_code, DIRECTION.FOUR_FIVE]
                self.assertEqual(looked_up_five_code, five_code)
            except ConversionError as e:
                if isinstance(e, ConversionError) and four_code in self.cnv.SPLIT_SPECIES:
                    # Great, wanted the exception:
                    pass
                else:
                    raise e


    #------------------------------------
    # test_from_six
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_six(self):

        # Test everying about the six-letter 
        # from-direction:
        
        self.assertEqual(self.cnv['PIOSEN', DIRECTION.SIX_FOUR], 'WCPA')
        self.assertEqual(self.cnv['PIOSEN', DIRECTION.SIX_SCI], 'PIONUS SENILIS')
        self.assertEqual(self.cnv['piosen', DIRECTION.SIX_FOUR], 'WCPA')

        # Test keys()       
        keys = list(self.cnv.keys(DIRECTION.SIX_FOUR))
        self.assertEqual(len(keys[0]), 6)

        # Test values()
        vals = list(self.cnv.values(DIRECTION.SIX_FOUR))
        self.assertEqual(len(vals[0]), 4)

        # Test items()
        for key, val in self.cnv.items(DIRECTION.SIX_FOUR):
            self.assertEqual(len(key), 6)
            self.assertEqual(len(val), 4)
            break

        for key, val in self.cnv.items(DIRECTION.SIX_SCI):
            self.assertEqual(len(key), 6)
            self.assertGreater(len(val), 6)
            break

    #------------------------------------
    # test_from_sci
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_sci(self):

        # Test everying about the scientific name
        # from-direction:
        self.assertEqual(self.cnv['Pionus senilis', DIRECTION.SCI_FOUR], 'WCPA')
        self.assertEqual(self.cnv['PIONUS SENILIS', DIRECTION.SCI_FOUR], 'WCPA')
        self.assertEqual(self.cnv['PIONUS_SENILIS', DIRECTION.SCI_FOUR], 'WCPA')
        self.assertEqual(self.cnv['pionus_senilis', DIRECTION.SCI_FOUR], 'WCPA')

        # Test keys()        
        keys = list(self.cnv.keys(DIRECTION.SCI_FOUR))
        self.assertGreater(len(keys[0]), 6)

        # Test values()
        vals = list(self.cnv.values(DIRECTION.SCI_FOUR))
        self.assertEqual(len(vals[0]), 4)

        # Test items()
        for key, val in self.cnv.items(DIRECTION.SCI_FOUR):
            self.assertGreater(len(key), 6)
            self.assertEqual(len(val), 4)
            break

        for key, val in self.cnv.items(DIRECTION.SCI_SIX):
            self.assertGreater(len(key), 6)
            self.assertEqual(len(val), 6)
            break
        
        
    #------------------------------------
    # test_canonicalize_nm
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_canonicalize_nm(self):
        
        self.assertEqual(self.cnv.canonicalize_nm('fooo'), 'FOOO')
        self.assertEqual(self.cnv.canonicalize_nm('baaaar'), 'BAAAAR')
        self.assertEqual(self.cnv.canonicalize_nm('scientific name'), 'SCIENTIFIC NAME')
        self.assertEqual(self.cnv.canonicalize_nm('scientific_name'), 'SCIENTIFIC NAME')
        self.assertEqual(self.cnv.canonicalize_nm('Scientific name'), 'SCIENTIFIC NAME')


# ----------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()