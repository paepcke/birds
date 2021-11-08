'''
Created on Nov 5, 2021

@author: paepcke
'''
import argparse
from enum import Enum
import os
import sys

from bisect import bisect_left

from experiment_manager.experiment_manager import ExperimentManager

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from powerflock.matplotlib_crosshair_cursor import CrosshairCursor

from powerflock.power_member import PowerMember, PowerQuantileClassifier
from powerflock.signal_analysis import SignalAnalyzer
from data_augmentation.utils import Utils, Interval

class Action(Enum):
    GRID_SEARCH = 0
    TEST = 1
    VIZ_PROBS = 2 

class PowerEvaluator:
    '''
    classdocs
    '''
    THRESHOLD = 0.9   # Probability threshold
    SLIDE_WIDTH = 0.2 # sec
    SIG_ID = 3
    
    TRUTH_RECT_HEIGHTS = 500 # Hz
    '''Height of rectangles that indicate location of true bird calls in viz_probs'''
    
    #------------------------------------
    # Constructor
    #-------------------


    def __init__(self,
                 experiment_name,
                 species, 
                 actions,
                 sig_source_recording=None,
                 sig_source_sel_tbl=None,
                 test_recording=None,
                 test_sel_tbl=None
                 ):
        '''
        Constructor
        '''

        self.species = species
        self.actions = actions
        self.sig_source_recording = sig_source_recording
        self.sig_source_sel_tbl = sig_source_sel_tbl
        self.test_recording = test_recording
        self.test_sel_tbl = test_sel_tbl
        
        self._init_data_paths()

        if sig_source_recording is None:
            self.sig_source_recording = self.sel_rec_cmto_xc1
        if sig_source_sel_tbl is None:
            self.sig_source_sel_tbl = self.sel_tbl_cmto_xc1
            
        # Cache of loaded audio for visualizations
        # maps fileName --> np_array
        self.audio_dict = {}
        # Map audioFile to Raven spectrogram df
        self.spectro_dict = {}

        
        experiments = []
        experiments.append(
            ExperimentManager(os.path.join(self.experiment_dir, 
                                           experiment_name)))
        power_member = PowerMember(
            species_name=species, 
            spectral_template_info=zip([sig_source_recording], [sig_source_sel_tbl]),
            the_slide_width_time=self.SLIDE_WIDTH,
            experiment=experiments[0]
            )

        for action in actions:
            if action == Action.GRID_SEARCH:
                grid_res = self.grid_search()
            elif action == Action.TEST:
                pwr_res = self.run_test(power_member,
                                        self.test_recording,
                                        self.test_sel_tbl,
                                        self.SIG_ID,
                                        self.THRESHOLD
                                        )
            elif action == Action.VIZ_PROBS:
                ax = self.viz_probs(power_member,
                                    self.test_recording,
                                    self.test_sel_tbl,
                                    self.SIG_ID
                                    )

        print("Done")

    #------------------------------------
    # run_test
    #-------------------
    
    def run_test(self,
                 pwr_member,
                 rec_path,
                 sel_tbl_path,
                 sig_id,
                 thres,
                 
                 ):

        pwr_res = pwr_member.compute_probabilities(rec_path)

        # Add a Truth column to the result:
        pwr_res.add_overlap_and_truth(sel_tbl_path)

        clf = PowerQuantileClassifier(sig_id, thres)
        clf.fit(pwr_res)
        score = clf.score(pwr_res.probabilities(sig_id), 
                          pwr_res.truths(sig_id),
                          name=f"({self.SIG_ID}/{self.THRESHOLD}/{self.SLIDE_WIDTH})"
                          )
        return score

    #------------------------------------
    # grid_search
    #-------------------
    
    def grid_search(self):
        
        pwr_member = PowerMember(self.species, self.template)
        
        # pr_disp = sklearn.metrics.PrecisionRecallDisplay.from_estimator(
        #     clf,
        #     pwr_res.prob_df.probability,
        #     pwr_res.prob_df.Truth
        #     )
        # pr_disp.plot()
        from timeit import default_timer as timer

        start = timer()
        # 1hr:20min
        grid_res = PowerQuantileClassifier.grid_search(
            pwr_member,
            audio_file=self.sel_rec_cmto_xc1,
            selection_tbl_file=self.sel_tbl_cmto_xc1,
            sig_ids=np.arange(1.0, len(self.templates[0].signatures)), 
            quantile_thresholds=[0.80, 0.85, 0.90, 0.99],
            slide_widths=[0.05, 0.1, 0.2]
            )
        end = timer()
        print(end - start)
        print(grid_res)
        
# ---------------- Visualization -------------


    #------------------------------------
    # viz_probs
    #-------------------
    
    def viz_probs(self, 
                  power_member,
                  test_recording,
                  test_sel_tbl,
                  sig_id
                  ):
        
        if not power_member.output_ready:
            pwr_res = power_member.compute_probabilities(test_recording)
        else:
            pwr_res = power_member.power_member

        try:
            truths = pwr_res.truths(sig_id)
        except IndexError:
            # Nobody has told this power result about
            # what is true: 
            # Add a Truth column to the result:
            pwr_res.add_overlap_and_truth(test_sel_tbl)
            truths = pwr_res.truths(sig_id)
        
        try:
            spectro = self.spectro_dict[test_recording]
        except KeyError:
            spectro = SignalAnalyzer.raven_spectrogram(test_recording)
            self.spectro_dict[test_recording] = spectro
        
        mesh = plt.pcolormesh(spectro.columns, 
                              list(spectro.index), 
                              spectro, 
                              cmap='jet', 
                              shading='flat')
        
        ax = mesh.axes

        row_dicts = Utils.read_raven_selection_table(test_sel_tbl)
        call_intervals = [row_dict['time_interval']
                          for row_dict
                          in row_dicts
                          ]

        truth_rects_x = [interval['low_val']
                         for interval
                         in call_intervals]
        truth_rects_y = [0]*len(call_intervals)
        widths        = [interval['high_val'] - interval['low_val']
                         for interval
                         in call_intervals]
        heights       = [self.TRUTH_RECT_HEIGHTS]*len(call_intervals)

        fig = ax.figure
        fig.show()
        
        for x,y,width,height in zip(truth_rects_x, truth_rects_y, widths, heights):
            ax.add_patch(Rectangle((x,y),
                                   width, height,
                                   facecolor=None,
                                   fill=False,
                                   edgecolor='black',
                                   hatch='*')
                                   )
        cursor = PowerInfoCursor(ax, pwr_res, truths, call_intervals)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        input("Press ENTER to finish: ")

# --------------  Utilities -----------------

    def _init_data_paths(self):
        
        self.cur_dir = os.path.dirname(__file__)
        proj_root = os.path.join(self.cur_dir, '../..')
        self.experiment_dir = os.path.join(proj_root, 'experiments/PowerSignatures')
                                  

        self.sound_data = os.path.join(self.cur_dir, 'tests/signal_processing_sounds')
        self.xc_sound_data = os.path.join(self.cur_dir, 'tests/signal_processing_sounds/XenoCanto')
        
        
        # Field Recordings
        self.BAFFG_data = os.path.join(self.sound_data, 'BAFFG')
        self.CCROC_data = os.path.join(self.sound_data, 'CCROC')
        
        self.BAFFG1_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-58028.mp3')
        self.BAFFG2_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-75982.mp3')
        self.BAFFG3_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-85219.mp3')
        
        self.CCROC1_rec = os.path.join(self.CCROC_data, 'CALL_XC332432-ClayColoredThrush_Yucatan_081216_call2.mp3')
        self.CCROC2_rec = os.path.join(self.CCROC_data, 'CALL_XC482432-R028_Clay_coloured_thrush.mp3')
        self.CCROC3_rec = os.path.join(self.CCROC_data, 'CALL_XC540584-MixPre-255_Turdus_grayi.mp3')
        
        self.DCFLC_rec_fld      = os.path.join(self.sound_data, 'Field/DCFLC/DS_AM17_20190713_172958.WAV')
        self.DCFLC_sel_tbl_fld  = os.path.join(self.sound_data, 'Field/DCFLC/JZ_DS_AM17_20190713_172958.Table.1.selections.txt')
        
        self.sel_tbl_fld = os.path.join(self.cur_dir, 'tests/selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        # Full field recording for selection tbl: 
        self.sel_recording_fld = os.path.join(self.sound_data, 'DS_AM03_20190713_055956.wav')

        # Xeno Canto 
        self.sel_tbl_cmto_xc1 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto1.selections.txt')
        self.sel_rec_cmto_xc1 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        self.sel_tbl_cmto_xc2 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto2.selections.txt')
        self.sel_rec_cmto_xc2 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        self.sel_tbl_cmto_xc3 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto3.selections.txt')
        self.sel_rec_cmto_xc3 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')
        
        # Create signature templates to work with:
        self.recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        self.sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        self.templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                                 self.recordings, 
                                                                 self.sel_tbls)
# ---------------- Class PowerInfoCursor --------

class PowerInfoCursor(CrosshairCursor):
    
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, ax, pwr_res, truths, call_intervals):
        CrosshairCursor.__init__(self, ax)
        
        ax.texts[0].set_position((0.08, 0.8))
        self.pwr_res = pwr_res
        self.truths = truths
        self.call_intervals = call_intervals
        
        # Get number of signatures:
        self.num_sigs = len(pwr_res.prob_df.groupby(by='sig_id'))

    #------------------------------------
    # on_mouse_move
    #-------------------
    
    def on_mouse_move(self, event):
        x, y = event.xdata, event.ydata
        super().on_mouse_move(event)

        # Entering and exiting the image 
        # delivers y as None; avoid error for that:
        if x is None or y is None:
            return
        
        is_call = Interval.binary_search(self.call_intervals, x) > -1
        # For each signature, the current probability:
        probs_by_sig = []
        for sig_idx in range(self.num_sigs):
            probs_ser = self.pwr_res.probabilities(sig_idx+1)
            prob_times = probs_ser.index.values
            prob_idx = bisect_left(prob_times, x)
            probs_by_sig.append(probs_ser.iloc[prob_idx].round(2))
        
        txt = (f"time={x.round(2)}, freq={int(y)}\n"
               f"is call: {is_call}\n"
               f"probs by sig: {probs_by_sig}"
               )
        
        self.text.set_text(txt)
        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.horizontal_line)
        self.ax.draw_artist(self.vertical_line)
        self.ax.draw_artist(self.text)
        self.ax.figure.canvas.blit(self.ax.bbox)

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Evaluate or calibrate the power signature approach"
                                     )

    parser.add_argument('--sig_rec',
                        help='path to recording from which signatures are to be computed',
                        default=None)
    parser.add_argument('--sig_sel',
                        help='path to selection table for signature',
                        default=None)
    parser.add_argument('--test_rec',
                        help='path to recording that is to be tested against a given signature',
                        default=None)
    parser.add_argument('--test_sel',
                        help='path to recording selection table for recording',
                        default=None)
    parser.add_argument('-s', '--species',
                        help='species for which action should be performed',
                        default=None)
    parser.add_argument('-e', '--experiment_name',
                        help='name for root dir of experiment where to save results; default: Species name',
                        default=None)
    parser.add_argument('actions',
                        choices=['test', 'gridSearch', 'viz_probs'],
                        nargs='+',
                        help='Repeatable: actions to perform')

    args = parser.parse_args()

    # Check file existence:

    if args.sig_rec is not None and not os.path.exists(args.sig_rec):
        raise FileNotFoundError(f"Audio file for signature creation not found ({args.sig_rec})")
    if args.sig_sel is not None and not os.path.exists(args.sig_sel):
        raise FileNotFoundError(f"Raven selection table file for signature creation not found ({args.sig_sel})")
    
    if args.test_rec is not None and not os.path.exists(args.test_rec):
        raise FileNotFoundError(f"Audio file for testing not found ({args.test_rec})")
    if args.test_sel is not None and not os.path.exists(args.test_sel):
        raise FileNotFoundError(f"Raven selection table file for testing not found ({args.test_sel})")

    # Convert text actions to Action enum members:
    actions = []
    for action in args.actions:
        if action == 'test':
            actions.append(Action.TEST)
        elif action == 'gridSearch':
            actions.append(Action.GRID_SEARCH)
        elif action == 'viz_probs':
            actions.append(Action.VIZ_PROBS)
        else:
            raise NotImplementedError(f"Action {action} is not implemented")
    
    if args.experiment_name is None:
        args.experiment_name = args.species
    
    PowerEvaluator(args.experiment_name,
                   args.species,
                   actions,
                   sig_source_recording=args.sig_rec,
                   sig_source_sel_tbl=args.sig_sel,
                   test_recording=args.test_rec,
                   test_sel_tbl=args.test_sel
                   )
    print('foo')