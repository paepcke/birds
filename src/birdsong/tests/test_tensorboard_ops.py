'''
Created on Feb 11, 2021

@author: paepcke
'''
import os
import shutil
import sys
import unittest

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional_pil import to_grayscale

from birdsong.utils.tensorboard_manager import TensorBoardManager
# Use summary writer in tensorboard_plotter, because
# its add_hparams does not introduce a new directory: 
from birdsong.utils.tensorboard_plotter import SummaryWriterPlus
from birdsong.utils.tensorboard_plotter import TensorBoardPlotter


#********TEST_ALL = True
TEST_ALL = False

class TensorBoardOpsTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tensorboard_dir = os.path.join(cls.curr_dir, 'tensorboard_test_runs')
        os.makedirs(cls.tensorboard_dir)

        cls.bmw_images_path = os.path.join(cls.curr_dir, 'data/cars/bmw')

        # Start a tensorboard server, and pop it
        # up in a browser window:

        cls.tb_man = TensorBoardManager(cls.tensorboard_dir)

        # Start tensorboard, and fill with some scalars
        cls.writer1, cls.writer2 = cls.build_basic_tensorboard_info()


    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):

        try:
            if cls.tb_man is not None:
                cls.tb_man.close()
        except AttributeError:
            # No db_man created during test(s)
            pass

        try:
            shutil.rmtree(cls.tensorboard_dir)
        except Exception:
            pass
        
        print("You need to close browser windows/tabs manually :-(")

# -------------- Tests ------------------

    #------------------------------------
    # test_same_image_each_experiment
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_same_image_each_experiment(self):

        # Can't assert anything, but make sure
        # these operations at least run
        bmw1_path = os.path.join(self.bmw_images_path, 'bmw1.jpg')
        self.add_image(self.writer1, 'cars', bmw1_path, step=0)
        self.add_image(self.writer2, 'cars', bmw1_path, step=0)

    #------------------------------------
    # test_animated_matrix 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_animated_matrix(self):
        
        tb_plotter = TensorBoardPlotter()
        for epoch in range(5):
            mat, class_names = self.make_confusion_matrix()
            tb_plotter.conf_matrix_to_tensorboard(self.writer1, 
                                                  mat,
                                                  class_names, 
                                                  step=epoch)
        
        user_prompt = "Browser after refreshing: is there a scrollable confusion matrix in the IMAGES tab?"
        if not self.query_yes_no(user_prompt):
            self.fail('First confusion matrix did not show')
        

        # Now add a second series of confusion matrices,
        # this time for Exp2:
        
        for epoch in range(5):
            mat, class_names = self.make_confusion_matrix()
            tb_plotter.conf_matrix_to_tensorboard(self.writer2, 
                                                  mat,
                                                  class_names, 
                                                  step=epoch)
        
        user_prompt = "After refreshing, did a second series of matrices apeared in IMAGES tab? "
        if not self.query_yes_no(user_prompt):
            self.fail('Second confusion matrix did not show')


    #------------------------------------
    # test_hparams 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_hparams(self):
        
        hparm_names     = {'lr' : 0.001, 
                           'optimizer' : 'Adam',
                           'batch_size' : 32}
        
        hparm_measures  = {'_loss_train' : 40,
                           '_val_train'  : 20,
                           '_accuracy'   : .83
                           }
        
        self.writer1.add_hparams(hparm_names, hparm_measures)

        num_files = self.count_tensorboard_files()
        # Adding hparams adds an event file just for 
        # the hparms:
        self.assertEqual(num_files, 5)
        
        user_prompt = "After refreshing, is there a new HPARAMS tab in browser?"
        if not self.query_yes_no(user_prompt):
            self.fail('New HPARAMS tab did not show, or no hparam entry')
        
        # Add hparams for experiment 2: 
        
        hparm_names     = {'lr' : 0.005, 
                           'optimizer' : 'RMSProp',
                           'batch_size' : 64}
        
        hparm_measures  = {'_loss_train' : 60,
                           '_val_train'  : 50,
                           '_accuracy'   : .50
                           }
        
        self.writer2.add_hparams(hparm_names, hparm_measures)

        num_files = self.count_tensorboard_files()
        # Adding hparams adds an event file just for 
        # the hparms:
        self.assertEqual(num_files, 6)

        user_prompt = "After refreshing, is there a second line for Exp2 in HPARAMS tab?"
        if not self.query_yes_no(user_prompt):
            self.fail('Second hparam line did not show')

    #------------------------------------
    # test_prec_recall_curves_binary 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_prec_recall_curves_binary(self):
        
        # Binary labels 0/1, len single-dim 100:
        labels = torch.randint(0, 2, (100,))  
        predictions = torch.rand(100)
        epoch = 0
        self.writer1.add_pr_curve('P/R Curve Binary Epoch 0',
                                  labels,
                                  predictions, 
                                  global_step=epoch 
                                  )
        user_prompt = "After refresh: is there a PR curve for Exp1?: "
        self.query_yes_no(user_prompt)

        # Add more curves for addtional epochs:
        num_epochs_total = 5
        # Already have the curve for epoch 0: 
        for epoch in range(1, num_epochs_total):
            labels = torch.randint(0, 2, (100,))  
            predictions = torch.rand(100)
            self.writer1.add_pr_curve(f"P/R Curve Binary Epoch {epoch}",
                                      labels,
                                      predictions, 
                                      global_step=epoch 
                                      )

        user_prompt = f"After refresh: are there more PR curves for Exp1, epochs 1-{num_epochs_total - 1}?: "
        self.query_yes_no(user_prompt)
            
        # Add same number of curves for Exp 2:

        # Already have the curve for epoch 0: 
        for epoch in range(0, num_epochs_total):
            labels = torch.randint(0, 2, (100,))  
            predictions = torch.rand(100)
            self.writer2.add_pr_curve(f"P/R Curve Binary Epoch {epoch}",
                                      labels,
                                      predictions, 
                                      global_step=epoch 
                                      )

        user_prompt = f"After refresh: {num_epochs_total} more PR curves for Exp2? "
        self.query_yes_no(user_prompt)
         
    #------------------------------------
    # test_prec_recall_curves_multi_class
    #-------------------
#*************** Next:    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_prec_recall_curves_multi_class(self):
        
        # Binary labels 0/1, len single-dim 100:
        labels = torch.randint(0, 2, (100,))  
        predictions = torch.rand(100)
        epoch = 0
        self.writer1.add_pr_curve('P/R Curve Binary Epoch 0',
                                  labels,
                                  predictions, 
                                  global_step=epoch 
                                  )
        user_prompt = "After refresh: is there a PR curve for Exp1?: "
        self.query_yes_no(user_prompt)

# ------------------- Utils -----------

    #------------------------------------
    # build_basic_tensorboard_info 
    #-------------------

    @classmethod
    def build_basic_tensorboard_info(cls):
        '''
        Create two experiments. Within each, place
        values tagged loss/train, loss/val. Dir struct
        on return:

            tb_root     
               Exp1
                  1-file
               Exp2
                  1-file

        '''
        
        exp1_dir = os.path.join(cls.tensorboard_dir, 'Exp1')
        exp2_dir = os.path.join(cls.tensorboard_dir, 'Exp2')
        
        writer1 = SummaryWriterPlus(exp1_dir)
        writer2 = SummaryWriterPlus(exp2_dir)
        
        for epoch in range(5):
            val1 = epoch + 1
            val2 = epoch + 2
            
            writer1.add_scalar('loss/train', val1, epoch)
            writer1.add_scalar('loss/val', val1-0.5, epoch)
            
            writer2.add_scalar('loss/train', val2, epoch)
            writer2.add_scalar('loss/val', val2-0.5, epoch)
            
        return writer1, writer2
    
    #------------------------------------
    # add_image 
    #-------------------
    
    def add_image(self, writer, tag, img_path, step=0):
        
        img_height = 200 # px
        img_width  = 400 # px

        the_transforms = [transforms.Resize((img_height, img_width))]
        if to_grayscale:
            the_transforms.append(transforms.Grayscale())
        the_transforms.append(transforms.ToTensor())

        img_transform = transforms.Compose(the_transforms)
        img = Image.open(img_path)
        img = img_transform(img).float()

        # A 10px frame around each img:
        #grid = make_grid(img, padding=10)
        #writer.add_image(tag, grid, step)
        
        writer.add_image(tag, img, step)

    #------------------------------------
    # count_tensorboard_files 
    #-------------------
    
    def count_tensorboard_files(self):

        files_and_dirs_now = os.walk(self.tensorboard_dir)
        num_files = sum([len(filenames) + len(dirnames)
                           for _dirpath, dirnames, filenames
                            in files_and_dirs_now])
        return num_files

    #------------------------------------
    # make_confusion_matrix 
    #-------------------
    
    def make_confusion_matrix(self):
        dim = (6,6)
        num_classes = dim[0]
        class_names = [f"class_{class_num}"
                       for class_num
                       in range(num_classes)
                       ]

        matrix = (10*torch.rand(dim)).sub(0.5).round().int()
        return matrix, class_names

    #------------------------------------
    # query_yes_no 
    #-------------------

    def query_yes_no(self, question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.
    
        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    
        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)
    
        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")
# ---------------- Main --------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
