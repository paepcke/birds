'''
Created on Mar 20, 2021

@author: paepcke
'''
import copy

from adjustText import adjust_text
import natsort

import matplotlib.pyplot as plt

from data_augmentation.utils import Utils

class ClassificationPlotter(object):
    '''
    classdocs
    '''
    AXIS_TITLE_SIZE = 14 # pt
    BEST_OP_PT_SIZE = 12 # pt
    BOP_LABEL       = 'Optimal operation'
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, params):
        '''
        Constructor
        '''

    #------------------------------------
    # chart_pr_curves 
    #-------------------
    
    @classmethod
    def chart_pr_curves(cls, curve_info):
        '''
        Creates and returns a pyplot figure instance
        with one PR curve for each class. The returned
        figure is ready for calling show(). 

        The curve_info is expected to be a single
        CurveSpecification instance, or a list of such
        instances. One curve is drawn for each instance.
        
        Each curve should show the achieved f1 for each 
        precision/recall pair for a single class: one-against-all.
         
        Each instance should contain at least a 'precisions'
        and a 'recalls' key with the precs and recs
        that define the curve.
        
        If a CurveSpecification contains the following
        additional optional info, those will be shown
        in the plot:
        
               'precisions' : precisions,
               'recalls'    : recalls,
               # Optionally:
               'thresholds' : list of probability/logit
                              threshold values used to create the points
                              that make up the curve
               'avg_prec'   : avg_precision,
               'mAP'        : mean average precision summarizing all pr_curves
               'best_op_pt' : a BestOperatingPoint instance
               
        
        If the thresholds are provided, they
        will be integrated into the figure as
        a comment, or caption.
        
        If best_op_pt is provided, it identifies the
        point in a curve, and the threshold associated
        with the curve that maximize f1:
        
        best_op_pt: 
            threshold' : <optimal decision probability value>
            f1'        : <f1 at the optimal threshold>
            prec'      : <precision at the optimal threshold>
            rec'       : <recall at the optimal threshold>
        
        :param curve_info: dict of info for drawing the curves.
            Keys are class names or IDs; the values are dicts
            as documented above.
        :type curve_info: {int : CurveSpecification}
        '''

        # Leave the original curve spec instances alone:
        curve_info_cpy = copy.deepcopy(curve_info)
        # We don't need the class IDs that are the keys
        # of the curve_info dict. We want the curve specs
        # themselves.:
        
        curve_specs = list(curve_info_cpy.values()) 

        # The precision_recall_curve() uses different 
        # numbers of thresholds, each time (depending on 
        # when recall==1 is reached). To plot multiple
        # pr-curves in one fig, we need to make prec/rec
        # pd.Series lengths equal for all curves. Since
        # the precision_recall_curve() method fills the
        # precs and recs arrays with points from right to left,
        # we pad the *start* of the respective series with
        # their first point for all curves.
        
        # Find max-length precs pd.Series among the curves:
        prec_rec_lengths = [len(crv['precisions'])
                           for crv in curve_specs
                           ]
        max_prec_rec_len = max(prec_rec_lengths)

        for crv in curve_specs:
            padded_preds = Utils.pad_series(crv['precisions'], 'left', max_prec_rec_len)
            padded_recs = Utils.pad_series(crv['recalls'], 'left', max_prec_rec_len)
            crv['precisions'] = padded_preds
            crv['recalls'] = padded_recs

        fig = plt.figure()
        
        ax = fig.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Recall', size=cls.AXIS_TITLE_SIZE)
        ax.set_ylabel('Precision', size=cls.AXIS_TITLE_SIZE)

        # Possible point markers for each
        # successive curve:
        markers    = 'ox+*^@'
        ap_markers = '*'
        colors  = ['mediumblue', 'black', 'red', 'springgreen', 'magenta', 'chocolate']
        
        for class_label, curve_obj in enumerate(curve_specs):
            # Use plot idiom 
            #
            #  plot('<key_for_x_axis_data,
            #       '<key_for_y_axis_data,
            #       data=<dict>
            #
            # It's OK that curve_obj contains
            # other info than just recs/preds.
            # The [1:] selectors cut off the 
            # artificially introduced point that
            # forces the chart to start at 0. The
            # downside of this cutoff is that the 
            # X-axis does not end at 1, unless recall
            # goes that far. But w/o the cutoff the chart
            # is confusing:
            ax.plot(curve_obj['recalls'][1:],
                    curve_obj['precisions'][1:],
                    marker=markers[class_label],
                    color=colors[class_label],
                    label=f"Class {class_label}"
                    ) 
            # Highlight the optimal operating point:
            # if curve_obj has no BOP info, just
            # error out. Else: 'rec', 'pred' are the
            # keys for the recall/prediction values
            # of the BOP:
            try:
                bop_obj = curve_obj['best_op_pt']
                bop_xy   = (bop_obj['recall'], bop_obj['precision'])
                ax.plot(bop_xy[0], bop_xy[1],
                        color=colors[class_label],
                        marker=ap_markers,
                        markersize=cls.BEST_OP_PT_SIZE,
                        label=cls.BOP_LABEL
                        )
                # Next to the best op point, put
                # the corresponding threshold and
                # f1 value:
                bop_thresh = round(bop_obj['threshold'],2)
                bop_f1     = round(bop_obj['f1'], 2)

                bop_txt    = f"f1: {bop_f1}\n" + f"thresh: {bop_thresh}"
                ax.annotate(bop_txt, xy=bop_xy) 
                
            except KeyError:
                # No best operating point provided
                pass
            
            # Add the average precision (AP) and
            # the threshold that  into:
            try:
                avg_prec    = round(curve_obj['avg_prec'], 2)
                crv_txt     = f"AP: {avg_prec.round(2)}"
                # Left-most point of the curve:
                crv_xy_left = (curve_obj['recalls'][0],
                               curve_obj['precisions'][0]
                               )

                # Put AP and threshold info
                # at the left end of each curve:
                ax.annotate(crv_txt, crv_xy_left)

            except KeyError:
                # No average precisions available.
                pass


        handles, legend_labels = ax.get_legend_handles_labels()
        
        # There may be as many entries for 
        # the color of the best-operating-point
        # dot in the legend as there are classes
        # (i.e. curves). We only want one.

        bop_handle = None
        try:
            bop_idx = legend_labels.index(cls.BOP_LABEL)
            bop_handle = handles[bop_idx]

            # Lambda will receive each legend
            # (handle, label) tuple:
            filtered = \
              filter(lambda hand_lab_tup: hand_lab_tup[1] != cls.BOP_LABEL,
                     zip(handles, legend_labels))

            # The filtered var is an iterator
            # feeding out [(h1,l1), (h2,l2),...]
            # Unzip those into [(h1,h2,...), (l1,l2,...)]
            # The '*' does the splitting:
            handles, legend_labels = list(zip(*filtered))

        except ValueError:
            # No BOP was drawn in the above loop.
            # So, nothing to fix:
            pass
        
        # Now have legend entries without
        # a BOP entry. Sort them:
        legend_labels, handles = \
           zip(*natsort.natsorted(zip(legend_labels, handles), 
                                      key=lambda lab_hndl_tple: lab_hndl_tple[0]))

        # Did we take out BOP legend entries?
        
        if bop_handle is not None:
            # Turn the legend entry tuples into arrays
            # so we can manipulate the content:
            handles = list(handles)
            legend_labels = list(legend_labels)
            handles.append(bop_handle)
            legend_labels.append(cls.BOP_LABEL)

        
        ax.legend(handles, legend_labels, loc='lower left')

        
        # Give the figure a title
        fig.suptitle("Precision-Recall Plot")
        
        # Adjust text to avoid chart elements.
        # Of the many optional args there is surely 
        # one that multiplies distance by three.
        # Just adjust three times for now:
        adjust_text(ax.texts)
        adjust_text(ax.texts)
        adjust_text(ax.texts)
        return fig
    
# ---------------------- Utils -------------

#     #------------------------------------
#     # annotate_pt 
#     #-------------------
#     
#     @classmethod
#     def annotate_pt(cls, ax, pt_coords, txt):
#         ax.annotate(txt, xy=pt_coords)


#     #------------------------------------
#     # add_curve_annotation
#     #-------------------
# 
#     @classmethod
#     def add_curve_annotation(cls, ax, curve_obj, txt):
#         '''
#         Given a dict with at least keys:
# 
#              o recalls   : list of x coordinates
#              o precisions: list of y coordinates
#              
#         :param ax:
#         :type ax:
#         :param curve_obj:
#         :type curve_obj:
#         :param txt:
#         :type txt:
#         '''
#         
#         precs       = curve_obj['precisions']
#         recs        = curve_obj['recalls'] 
#         left_crv_pt = (recs[0], precs[0])
#         ax.annotate(txt,
#                     xy=left_crv_pt
#                     ) 

'''
        # The precision_recall_curve() uses different 
        # numbers of thresholds, each time (depending on 
        # when recall==1 is reached). To plot multiple
        # pr-curves in one fig, we need to make prec/rec
        # pd.Series lengths equal for all curves. Since
        # the precision_recall_curve() method fills the
        # precs and recs arrays with points from right to left,
        # we pad the *start* of the respective series with
        # their first point for all curves.
        
        # Find max-length precs pd.Series among the curves:
        prec_rec_lengths = [len(crv['precisions'])
                           for crv in list(pr_curve_specs.values())
                           ]
        max_prec_rec_len = max(prec_rec_lengths)

        #***** Don't extend all the series.
        #      Only do it for the curves we actually
        #      plot******
        for crv in list(pr_curve_specs.values()):
            padded_preds = cls.pad_series(crv['precisions'], 'left', max_prec_rec_len)
            padded_recs = cls.pad_series(crv['recalls'], 'left', max_prec_rec_len)
            crv['predictions'] = padded_preds
            crv['recalls'] = padded_recs

'''