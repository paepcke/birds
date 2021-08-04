'''
Created on May 24, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import sys

from logging_service import LoggingService
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

from birdsong.nets import NetUtils
from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
from skorch import NeuralNet

class SaliencyMapper:
    '''
    classdocs
    '''
    # Sorting such that numbers in strings
    # do the right thing: "foo2" after "foo10": 
    # Should actually be 1:3 but broke the system:
    #SAMPLE_WIDTH  = 400 # pixels
    #SAMPLE_HEIGHT = 400 # pixels
    SAMPLE_WIDTH  = 215 # pixels
    SAMPLE_HEIGHT = 128 # pixels
    

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 config_info,
                 model_path, 
                 in_img_or_dir,
                 gpu_to_use=0,
                 unittesting=False):
        '''
        Constructor
        '''
        if unittesting:
            return

        self.log = LoggingService()
        self.samples_path = in_img_or_dir
        
        try:
            self.config = Utils.read_configuration(config_info)
        except Exception as e:
            msg = f"During config init: {repr(e)}"
            self.log.err(msg)
            raise RuntimeError(msg) from e
        
        self.num_classes = self.config.getint('Testing', 'num_classes')
        dataloader = SaliencyDataloader(in_img_or_dir, self.config)
        
        self.prep_model_inference(model_path, gpu_to_use)

        for img, metadata in dataloader:
            saliency_img = self.create_one_saliency_map(img, metadata)

    #------------------------------------
    # create_one_saliency_map
    #-------------------
    
    def create_one_saliency_map(self, img, metadata):

        # We must run the model in evaluation mode
        self.model.eval()
    
        try:
            species = metadata['species']
        except KeyError:
            self.log.warn(f"Image has no species metadata")
            species = None
        
        img.requires_grad_()
        
        # Forward pass through the model to get the 
        # scores 

        scores = self.model(img)

        # Get the index corresponding to the maximum score and the maximum score itself.
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        

        # Backward function on score_max performs 
        # the backward pass in the computation graph and 
        # calculates the gradient of score_max with respect 
        # to nodes in the computation graph:

        score_max.backward()
    
        # Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
        # R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
        # across all colour channels.

        saliency, _ = torch.max(img.grad.data.abs(),dim=1)
        
        # code to plot the saliency map as a heatmap
        plt.imshow(saliency[0], cmap=plt.cm.hot)
        plt.axis('off')
        plt.show()

    #------------------------------------
    # prep_model_inference 
    #-------------------

    def prep_model_inference(self, model_path, gpu_to_use):
        '''
        1. Parses model_path into its components, and 
            creates a dict: self.model_props, which 
            contains the network type, grayscale or not,
            whether pretrained, etc.
        2. Creates self.csv_writer to write results measures
            into csv files. The destination file is determined
            as follows:
                <script_dir>/runs_raw_inferences/inf_csv_results_<datetime>/<model-props-derived-fname>.csv
        3. Creates self.writer(), a tensorboard writer with destination dir:
                <script_dir>/runs_inferences/inf_results_<datetime>
        4. Creates an ImageFolder classed dataset to self.samples_path
        5. Creates a shuffling DataLoader
        6. Initializes self.num_classes and self.class_names
        7. Creates self.model from the passed-in model_path name
        
        :param model_path: path to model that will be used for
            inference by this instance of Inferencer
        :type model_path: str
        '''
        
        self.model_path = model_path
        model_fname = os.path.basename(model_path)
        
        # Extract model properties
        # from the model filename:
        self.model_props  = FileUtils.parse_filename(model_fname)
        
        # Get the right type of model,
        # Don't bother getting it pretrained,
        # of freezing it, b/c we will overwrite 
        # the weights:
        
        self.model = NetUtils.get_net(
            self.model_props['net_name'],
            num_classes=self.num_classes,
            pretrained=False,
            freeze=0,
            to_grayscale=self.model_props['to_grayscale']
            )

        sko_net = NeuralNet(
            module=self.model,
            criterion=torch.nn.NLLLoss,
            batch_size=1,
            train_split=None,    # We won't train, just infer
            #callbacks=[EpochScoring('f1')]
            device=f"cuda:{gpu_to_use}" if torch.cuda.is_available() else "cpu"
            )

        sko_net.initialize()  # This is important!
        sko_net.load_params(f_params=self.model_path)
        

# ---------------------- Dataset Class ---------------

class SaliencyDataloader:
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, img_or_dir_path, config, sample=False):


        self.SAMPLE_WIDTH  = config.getint('Training', 'sample_width')
        self.SAMPLE_HEIGHT = config.getint('Training', 'sample_height')
        # Collect img files into a list:

        self.files_to_show = []
        
        if os.path.isfile(img_or_dir_path):
            # Just a single sample file to process:
            if Path(img_or_dir_path).suffix.lower() in self.IMG_EXTENSIONS:
                self.files_to_show.append(img_or_dir_path)
        else:
            # Was given a directory; does it have only
            # (hopefully) species subdirectories?
            for root, _dirs, files in os.walk(img_or_dir_path):
                for file in files:
                    if Path(file).suffix.lower() in self.IMG_EXTENSIONS: 
                        self.files_to_show.append(os.path.join(root, file))
        
        # Reverse so we can use pop()
        # and preserve order of files:
        self.files_to_show.reverse()

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        return self

    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        '''
        Returns a two-tuple: image tensor, img metadata
        
        :return Image loaded as a PIL, then downsized,
            and transformed to a tensor. Plus any metadata
            the image contains
        :rtype (torch.Tensor, {str : str})
        '''
        
        try:
            fname = self.files_to_show.pop()
        except IndexError:
            raise StopIteration()
        img_obj_pil, metadata = SoundProcessor.load_spectrogram(fname, to_nparray=False)
        img_obj_tns = self.transform_img(img_obj_pil)
        #*****new_img_obj = self.transform_img(img_obj_np)
        img_obj_tns = torch.tensor(img_obj_tns).unsqueeze(dim=0)

        return img_obj_tns, metadata
        
    #------------------------------------
    # transform_img 
    #-------------------
    
    def transform_img(self, img_obj, to_grayscale=True):
        '''
        Given an image object as return by torchvision.Image.opent(),
        transform the img to conform to Birdsong project's
        standard dimensions and mean/std.
        
        :param img_obj: instance holding the in-memory
            data structure of a .png file
        :type img_obj: torchvision.Image
        :param to_grayscale: whether or not to convert
            to grayscale
        :type to_grayscale: bool
        :return: a new, transformed image
        :rtype: torchvision.Image
        '''
        
        is_grayscale = img_obj.mode == 'L'
        if to_grayscale and not is_grayscale:
            # Need to transform to grayscale:
            try:
                new_img = self._composed_transform_to_grayscale(img_obj)
                return new_img
            except AttributeError:
                # Haven't created _composed_transform_to_grayscale yet:
                xforms = self._standard_transform()
                xforms.append(transforms.Grayscale())
                self._composed_transform_to_grayscale = transforms.Compose(xforms)
                return self._composed_transform_to_grayscale(img_obj)
        else:
            
            try:
                new_img = self._composed_transform(img_obj)
                return new_img
            except AttributeError:
                # Not yet cached:
                self._composed_transform = transforms.Compose(self._standard_transform())
                return self._composed_transform(img_obj)

# ------------------ Utilities -----------------

    #------------------------------------
    # _standard_transform 
    #-------------------
    
    def _standard_transform(self):
        
        # The input images are already normalized
        # and grayscale in our workflow, so the 
        # usual 3-channel normalization transform is
        # commented out:
        img_transforms = [transforms.Resize((self.SAMPLE_WIDTH, self.SAMPLE_HEIGHT)),
                          transforms.ToTensor(),
                          #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          #                     std=[0.229, 0.224, 0.225])
                          ]
        return img_transforms


# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Show images of where a given model is 'looking'."
                                     )

    parser.add_argument('-g', '--gpu',
                        type=int,
                        help='index number of GPU to use; default: 0, or CPU',
                        default=0)

    parser.add_argument('config_info',
                        help='path to config file',
                        default=None)

    parser.add_argument('model',
                        help='path to a trained model',
                        default=None)

    parser.add_argument('input',
                        help='path to an image, or root directory of images',
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input image or directory not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    gpu = args.gpu if torch.cuda.is_available() else 'cpu'
    SaliencyMapper(args.config_info, args.model, args.input, gpu_to_use=gpu)
