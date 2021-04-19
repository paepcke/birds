'''
Created on Mar 21, 2021

@author: paepcke
'''
import unittest

import matplotlib.pyplot as plt

from birdsong.ml_plotting.classification_charts import ClassificationPlotter
from birdsong.utils.utilities import FileUtils
import numpy as np


TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    #------------------------------------
    # test_chart_pr_curves 
    #-------------------
    
    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_chart_pr_curves(self):
        recall_axis = np.array([1,2,3,4,5,6])
        curve_info = {1 : {'recalls'    : recall_axis,
                           'precisions' : 2 * recall_axis,
                           'avg_prec'   : 0.6,
                           'best_op_pt' : {'threshold' : 0.6,
                                           'f1' : 0.82,
                                           'rec'  : 2,
                                           'prec' : 4,
                                           }
                           },
                      2 : {'recalls'    : recall_axis,
                           'precisions' : 0.5 * recall_axis,
                           'avg_prec'   : 0.8,
                         }
                     }
    
        num_classes, fig = \
            ClassificationPlotter.chart_pr_curves(curve_info)
            
        self.assertEqual(num_classes, 2)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'recall')
        self.assertEqual(ax.get_ylabel(), 'precision')

        # Allow the fig to show
        # before asking user to
        # check it (the pause())
        fig.show()
        plt.pause(0.001)
        
        fig_ok = FileUtils.user_confirm(f"Fig should have 2 lines, one point, and a legend\n" +\
                                         f"Looks OK? (Y/n")
        if not fig_ok:
            self.fail("PR curve was not correct")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
