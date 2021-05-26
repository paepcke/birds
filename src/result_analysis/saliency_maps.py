'''
Created on May 24, 2021

@author: paepcke
'''
import argparse
import os
import sys

from matplotlib import pyplot as plt
import torch
from torchvision import transforms

from birdsong.nets import NetUtils
from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils


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
                 model_path, 
                 in_img_or_dir, 
                 gpu_to_use=0,
                 unittesting=False):
        '''
        Constructor
        '''
        if unittesting:
            return
        
        self.transform_img = self.create_img_transform()
        self.model = self.materialize_model(model_path, gpu_to_use=gpu_to_use)
        img_gen = self.img_generator(in_img_or_dir)
        for img_path in img_gen:
            saliency_img = self.create_one_saliency_map(img_path, self.model)

    #------------------------------------
    # create_dataset
    #-------------------
    
    def img_generator(self, in_img_or_dir):
        
        if os.path.isfile(in_img_or_dir):
            return iter([in_img_or_dir])
        return Utils.find_in_tree_gen(in_img_or_dir, '*.png')

    #------------------------------------
    # materialize_model
    #-------------------
    
    def materialize_model(self, model_path, gpu_to_use=0):

        model_fname = os.path.basename(model_path)
        
        # Extract model properties
        # from the model filename:
        self.model_props  = FileUtils.parse_filename(model_fname)
        model = NetUtils.get_net(
            self.model_props['net_name'],
            num_classes=self.model_props['num_classes'],
            pretrained=False,
            freeze=0,
            to_grayscale=self.model_props['to_grayscale']
            )

        try:
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(self.model_path))
                FileUtils.to_device(model, 'gpu', gpu_to_use)
            else:
                self.model.load_state_dict(torch.load(
                    model_path,
                    map_location=torch.device('cpu')
                    ))
        except RuntimeError as e:
            emsg = repr(e)
            if emsg.find("size mismatch for conv1") > -1:
                emsg += " Maybe model was trained with to_grayscale=False, but local net created for grayscale?"
                raise RuntimeError(emsg) from e

        return model
        
    #------------------------------------
    # create_one_saliency_map
    #-------------------
    
    def create_one_saliency_map(self, model, img_path):

        # We must run the model in evaluation mode
        model.eval()
    
        img_arr, metadata = SoundProcessor.load_spectrogram(img_path)
    
        img.requires_grad_()
        
        # Forward pass through the model to get the 
        # scores 

        scores = model(img)
        
    
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

    #------------------------------------
    # load_img 
    #-------------------
    
    def load_img(self, img_path):
        '''
        Returns a two-tuple: image tensor, img metadata
        
        :param img_path: full path to image
        :type img_path: str
        :return Image loaded as a PIL, then downsized,
            and transformed to a tensor. Plus any metadata
            the image contains
        :rtype (torch.Tensor, {str : str})
        '''

        img_obj_np, metadata = SoundProcessor.load_spectrogram(img_path, to_nparray=True)
        img_obj_tns = torch.tensor(img_obj_np).unsqueeze(dim=0)
        new_img_obj = self.transform_img(img_obj_tns)
        print('bar')

        #****return (img_tensor, torch.tensor(label))
        
# ------------------ Utilities -----------------

    #------------------------------------
    # _standard_transform 
    #-------------------
    
    def _standard_transform(self):
        
        img_transforms = [transforms.Resize((self.SAMPLE_WIDTH, self.SAMPLE_HEIGHT)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                          ]
        return img_transforms

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Show images of where a given model is 'looking'."
                                     )

    parser.add_argument('-g' '--gpu',
                        help='index number of GPU to use; default: 0',
                        default=0)

    parser.add_argument('input',
                        help='path to an image, or root directory of images',
                        default=None)
    parser.add_argument('model',
                        help='path to a trained model',
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input image or directory not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
        
    SaliencyMapper(args.input, args.model, gpu_to_use=args.gpu)
