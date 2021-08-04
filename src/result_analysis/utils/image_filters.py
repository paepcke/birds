'''
Created on Aug 4, 2021

@author: paepcke
'''
import argparse
import os
import sys

from scipy import ndimage
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
#from PIL import Image

class Filter:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, img_path, filter_name, outfile=None):
        '''
        Constructor
        '''
        
        if not os.path.exists(img_path):
            print(f"Cannot find {img_path})")
            
        #img = Image.open(img_path).convert("L")
        img = plt.imread(img_path)
        
        if filter_name == 'sobel':
            img_filtered = self.sobel_filter(img)
        elif filter_name == 'median':
            img_filtered = self.median_filter(img)
        else:
            raise NotImplementedError(f"Filter {filter_name} is not implemented")
            
        fig, _ax_before, ax_after = self.show_result(img, img_filtered)

        if outfile is not None:
            self.save_axes(fig, ax_after, outfile)
        
        plt.close('all')
        
    #------------------------------------
    # sobel_filter
    #-------------------

    def sobel_filter(self, img):
        
        result = ndimage.sobel(img)
        return result

    #------------------------------------
    # median_filter
    #-------------------
    
    def median_filter(self, img):
        
        result = medfilt2d(img)
        return result

    #------------------------------------
    # show_result 
    #-------------------
    
    def show_result(self, img, img_filtered):
        
        fig, (ax_before, ax_after) = plt.subplots(nrows=1, ncols=2)
        ax_before.axis('off')
        ax_after.axis('off')
        
        ax_before.imshow(img, cmap='gray')
        ax_after.imshow(img_filtered, cmap='gray')
        fig.show()
        return fig, ax_before, ax_after

    #------------------------------------
    # save_axes 
    #-------------------
    
    def save_axes(self, fig, ax, outfile):
        
        extent = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())        
        fig.savefig(outfile, bbox_inches=extent)

    #------------------------------------
    # full_extent 
    #-------------------
    
    def full_extent(self, ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels() 
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])
        return bbox.expanded(1.0 + pad, 1.0 + pad)


# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Here is what this package does."
                                     )

    parser.add_argument('-o', '--output',
                        help='path for result',
                        default=None)

    parser.add_argument('input',
                        help='fully qualified path to input image',
                        default=None)
    parser.add_argument('filter',
                        help='name of filter to apply')

    args = parser.parse_args()

    #*************
    args.filter='median'
    #*************    
    Filter(args.input, args.filter)
    #Filter(args.input, args.filter, outfile='/tmp/sobel_test.png')
