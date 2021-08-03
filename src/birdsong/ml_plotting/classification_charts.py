'''
Created on Mar 20, 2021

@author: paepcke
'''
import copy

from adjustText import adjust_text

import matplotlib.pyplot as plt
import pandas as pd

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
    def chart_pr_curves(cls, curve_specs, class_names):
        '''
        Creates and returns a pyplot figure instance
        with one PR curve for each class. The returned
        figure is ready for calling show(). 

        The curve_specs is expected to be a single
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
        
        :param curve_specs: info for drawing the curves.
        :type curve_specs: [CurveSpecification]
        '''
        # Leave the original curve spec instances alone:
        curve_specs = copy.deepcopy(curve_specs)

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

        new_data_series = {}
        for crv in curve_specs:
            new_data_series['Precision']  = Utils.pad_series(crv['precisions'], 'left', max_prec_rec_len)
            new_data_series['Recall']     = Utils.pad_series(crv['recalls'], 'left', max_prec_rec_len)
            new_data_series['Threshold']  = Utils.pad_series(crv['thresholds'], 'left', max_prec_rec_len)            
            new_data_series['f1']         = Utils.pad_series(crv['f1_scores'], 'left', max_prec_rec_len)            
            crv.pr_curve_df               = pd.DataFrame.from_dict(new_data_series) 

        # Make the figure 20% larger than 
        # default to allow for the legend 
        # being outside the chart axes:
        #def_fig_width, def_fig_height = rcParams['figure.figsize']
        fig = plt.figure(dpi=150.0, figsize=[9,5])
        
        # Single chart:
        ax = fig.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Recall', size=cls.AXIS_TITLE_SIZE)
        ax.set_ylabel('Precision', size=cls.AXIS_TITLE_SIZE)

        # Possible point markers for each
        # successive curve:
        markers    = 'ox+*^@'
        ap_markers = '*'
        colors  = ['mediumblue', 'black', 'red', 'springgreen', 'magenta', 'chocolate']
        
        chart_artists = {}
        for crv_idx, curve_obj in enumerate(curve_specs):
            
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
            class_name = class_names[curve_obj['class_id']]
            line_artist_arr = ax.plot(curve_obj['recalls'][1:],
                                      curve_obj['precisions'][1:],
                            	      marker=markers[crv_idx],
                            	      color=colors[crv_idx],
                                      label=f"Class {class_name}"
                        )
            # Highlight the optimal operating point:
            # if curve_obj has no BOP info, just
            # error out. Else: 'recall', 'precision' are the
            # keys for the recall/prediction values
            # of the BOP:
            try:
                bop_obj = curve_obj['best_op_pt']
                bop_xy   = (bop_obj['recall'], bop_obj['precision'])
                bop_artist_arr = ax.plot(bop_xy[0], bop_xy[1],
                                         color=colors[crv_idx],
                                         marker=ap_markers,
                                         markersize=cls.BEST_OP_PT_SIZE,
                                         label=f"Best OP pt {class_name}"
                                         )
                # Next to the best op point, put
                # the corresponding threshold and
                # f1 value:
                bop_thresh = round(bop_obj['threshold'],2)
                bop_f1     = round(bop_obj['f1'], 2)

                bop_txt    = f"f1: {bop_f1}\n" + f"thresh: {bop_thresh}"
                #********ax.annotate(bop_txt, xy=bop_xy) 
                
            except KeyError:
                # No best operating point provided
                pass
            
            # Remember the line and start artists:
            chart_artists[class_name] = (line_artist_arr[0], bop_artist_arr[0])
            
            # Add the average precision (AP) and
            # the threshold that generates the optimum
            # as an annotation to the star:
            try:
                avg_prec    = round(curve_obj['avg_prec'], 2)
                crv_txt     = f"AP: {avg_prec.round(2)}\n{bop_txt}"

                # Put AP and threshold info
                # on top of the AP marker; we'll move
                # it to the side later:
                ax.annotate(crv_txt, bop_xy)

            except KeyError:
                # No average precisions available.
                pass

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis,
        # centered half way down:
        _legend_obj = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
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