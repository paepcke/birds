#!/usr/bin/env python3
'''
Created on May 24, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys

from logging_service import LoggingService
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

from birdsong.nets import NetUtils
from birdsong.utils.tensorboard_plotter import TensorBoardPlotter
from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
from skorch import NeuralNet


class SaliencyMapper:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 model_path, 
                 in_img_or_dir,
                 config_info=None,
                 outdir=None,
                 gpu_to_use=0,
                 sample=0,
                 unittesting=False):
        '''
        Constructor
        '''
        if unittesting:
            return

        self.log = LoggingService()
        self.samples_path = in_img_or_dir

        # If no config file pt, assume it's in the usual place:
        if config_info is None:
            proj_root_dir = os.path.join(os.path.dirname(__file__), '../..')
            config_info = os.path.join(proj_root_dir, 'config.cfg')

        try:
            self.config = Utils.read_configuration(config_info)
        except Exception as e:
            msg = f"During config init: {repr(e)}"
            self.log.err(msg)
            raise RuntimeError(msg) from e
        
        self.num_classes = self.config.getint('Testing', 'num_classes')
        dataloader = SaliencyDataloader(in_img_or_dir, self.config, sample=sample)
        self.class_names = dataloader.class_names
        
        self.prep_model_inference(model_path, gpu_to_use)

        if outdir is not None:
            # Create dir if not exists:
            os.makedirs(outdir, exist_ok=True)
            
        for img, metadata in dataloader:
            saliency_fig, pred_class_id = self.create_one_saliency_map(img, metadata)

            if self.class_names is None or len(self.class_names) < pred_class_id:
                saliency_fig.text(.5,.1, f"Predicted: {pred_class_id}")
            else:
                saliency_fig.text(.5,.1, f"Predicted: {self.class_names[pred_class_id]}")

            saliency_fig.show()
            if outdir is not None:
                species_name = metadata['species']
                path_not_found = True
                i = 0
                while path_not_found:
                    out_path = os.path.join(outdir, f"{species_name}_{i}.pdf")
                    if os.path.exists(out_path):
                        i += 1
                        continue
                    path_not_found = False
                saliency_fig.savefig(out_path, format='pdf', dpi=150)
                
            to_do = input("Hit ENTER for next img; q to quit: ")
            plt.close(saliency_fig)
            if to_do == 'q':
                plt.close('all')
                return

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
        
        # Just for information: get winning prediction:
        probs = scores.softmax(dim=1).squeeze()
        pred_class_id = probs.argmax()

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
        saliency_img = saliency.detach().squeeze()
        img_3D = img.squeeze()
        if species is not None:
            # Only write species onto the image if we know it:
            img_with_species_4D = TensorBoardPlotter.print_onto_image(img_3D, species, (210,220))
            img_with_species_3D = img_with_species_4D.squeeze()
        else:
            img_with_species_3D = img_3D.detach()
         
        fig, (ax_saliency, ax_orig) = plt.subplots(nrows=1, ncols=2)
        
        ax_saliency.imshow(saliency_img, cmap=plt.cm.hot)
        ax_orig.imshow(img_with_species_3D, cmap='gray')

        ax_saliency.axis('off')
        ax_orig.axis('off')

        return fig, pred_class_id

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
    
    IMG_EXTENSIONS    = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    SHELL_SCRIPT_DIR  = os.path.join(os.path.dirname(__file__), 'shell_scripts')
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, img_or_dir_path, config, sample=0):

        if sample is not None and type(sample) != int:
            raise TypeError(f"Sample argument must be None or int, not {sample}")

        self.sample = sample if sample > 0 else None

        self.SAMPLE_WIDTH  = config.getint('Training', 'sample_width')
        self.SAMPLE_HEIGHT = config.getint('Training', 'sample_height')
        # Collect img files into a list:

        self.files_to_show = []
        
        if os.path.isfile(img_or_dir_path):
            # Just a single sample file to process:
            if Path(img_or_dir_path).suffix.lower() in self.IMG_EXTENSIONS:
                self.files_to_show.append(img_or_dir_path)
            else:
                raise TypeError(f"File {img_or_dir_path} does not have a known image extension")
                
        elif self.sample is None:
            # Collect all the image files below given dir:
            for root, _dirs, files in os.walk(img_or_dir_path):
                for file in files:
                    if Path(file).suffix.lower() in self.IMG_EXTENSIONS: 
                        self.files_to_show.append(os.path.join(root, file))
            # Reverse so we can use pop()
            # and preserve order of files:
            self.files_to_show.reverse()

        else:
            # We are to show a sample of images from each species 
            # below the directory.
            # In this case, make reasonably sure that we are pointing to
            # the root of a species tree:
            
            if not(self._is_species_root_dir(img_or_dir_path)):
                raise ValueError("Can sample images only from species root directory")
            
            # Go through each subdir, and pick a random
            # sample of images. Keep the full paths as 
            # lists, separately by species:
            
            sample_paths = {}
            species_dirs = Utils.listdir_abs(img_or_dir_path)

            self.class_names = [Path(full_path).stem for full_path in species_dirs] 
            
            for species_dir in species_dirs:
                species   = Path(species_dir).stem
                all_files = Utils.listdir_abs(species_dir)
                num_files = len(all_files)
                if num_files == 0:
                    sample_paths['species'] = []
                    continue
                # Fewer images than number of samples
                # requested? 
                if num_files <= self.sample:
                    # Yep, just use the ones we have:
                    sample_paths[species] = all_files
                    continue
                # Do the sampling for one species subdir:
                sample_paths[species] = [all_files[idx]
                                         for idx
                                         in random.sample(range(0,num_files), self.sample)
                                         ]
            self.sample_paths = sample_paths

    #------------------------------------
    # _is_species_root_dir 
    #-------------------
    
    def _is_species_root_dir(self, path):
        '''
        Takes a guess whether the given path is
        the root of a species directories tree.
        I.e. whether:
        
           path
               SPECIES1_DIR
                   fname.<img-extension>
               SPECIES2_DIR
                   fname.<img-extension>
                   ...
                   
        Ensures that path only includes directories,
        and checks that the first file in each directory
        is an image file.
                   
        :param path: absolute path to test
        :type path: src
        :return True if path is likely to be the root
            of a species tree, else False
        :rtype bool
        '''
        
        if os.path.isfile(path):
            return False
        
        # All files in the given dir must
        # themselves be dirs for path to 
        # be a species root:
        root, dirs, files = next(os.walk(path))
        if len(files) > 0:
            return False
        
        # Check each of the directories
        # to ensure that it is either empty,
        # or the first file is an image file:
        
        for the_dir in dirs:
            species_dir = os.path.join(root, the_dir)
            first_file = self._call_bash_script('first_file.sh', species_dir)
            if len(first_file) == 0 or Path(first_file).suffix in self.IMG_EXTENSIONS:
                continue
            else:
                return False

        return True

    #------------------------------------
    # _call_bash_script 
    #-------------------
    
    def _call_bash_script(self, script, *args):
        
        if not os.path.isabs(script):
            script_path = os.path.join(self.SHELL_SCRIPT_DIR, script)
        call_parms = [script_path] + list(args)
        proc = subprocess.run(call_parms, capture_output=True)
        res = proc.stdout.strip().decode("utf-8")
        return res

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        if self.sample:
            self.species_rotation = 0
            self.species = list(self.sample_paths.keys())
        return self

    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        '''
        Returns a two-tuple: image tensor, img metadata
        Behavior depends on whether we are feeding out
        a complete list of image files, or were asked to
        sample from multiple species. 
        
        If the latter, alternately feed out from the
        different species, which are in the self.sample_paths
        dict. 
        
        :return Image loaded as a PIL, then downsized,
            and transformed to a tensor. Plus any metadata
            the image contains
        :rtype (torch.Tensor, {str : str})
        '''
        
        if self.sample:
            have_fname = False
            while not have_fname:
                species_to_serve = self.species[self.species_rotation]
                try:
                    fname = self.sample_paths[species_to_serve].pop()
                    have_fname = True
                    # Prepare for the next call to __next__():
                    self.species_rotation += 1
                except IndexError:
                    # No more samples for this species.
                    # Try the next:
                    self.species_rotation += 1
                    if self.sample_rotation >= len(self.sample_paths):
                        # Exhausted the image samples of
                        # all species:
                        raise StopIteration()

        else:
            # No sampling, just a fixed set of image file
            # in self.files_to_show:
            try:
                fname = self.files_to_show.pop()
            except IndexError:
                raise StopIteration()

        img_obj_pil, metadata = SoundProcessor.load_spectrogram(fname, to_nparray=False)
        img_obj_tns_almost = self.transform_img(img_obj_pil)
        img_obj_tns = img_obj_tns_almost.clone().detach().unsqueeze(dim=0)

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
                # If the image is RGBA, then the conversion to
                # grayscale fails. In that case, convert to RGB
                # first:
                if img_obj.mode == 'RGBA':
                    img_obj = img_obj.convert(mode='RGB')
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
    
    parser.add_argument('-s', '--sample',
                        type=int,
                        help='number of samples to take from each species; default: no sampling',
                        default=0)
    
    parser.add_argument('-o', '--outdir',
                        help='directory for saving saliency figures',
                        default=None)

    parser.add_argument('-c', '--config_info',
                        help='path to config file; default: config.cfg in proj root',
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
    SaliencyMapper(args.model, 
                   args.input,
                   outdir=args.outdir,
                   config_info=args.config_info,
                   sample=args.sample,
                   gpu_to_use=gpu)
