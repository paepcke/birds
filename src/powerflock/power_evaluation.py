#!/usr/bin/env python
'''
Created on Nov 5, 2021

@author: paepcke
'''
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta
from enum import Enum
import os
import sys

from experiment_manager.experiment_manager import ExperimentManager
from logging_service.logging_service import LoggingService
from matplotlib.patches import Rectangle

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import Utils, Interval
import matplotlib.pyplot as plt
import numpy as np
from powerflock.matplotlib_crosshair_cursor import CrosshairCursor
from powerflock.power_member import PowerMember, PowerQuantileClassifier, \
    PowerResult
from powerflock.quad_sig_calibration import QuadSigCalibrator
from powerflock.signal_analysis import SignalAnalyzer

class Action(Enum):
    UNITTEST = 0
    GRID_SEARCH = 1
    TEST = 2
    VIZ_PROBS = 3 

class PowerEvaluator:
    '''
    classdocs
    '''
    THRESHOLD = 0.9   # Probability threshold
    SLIDE_WIDTH = 0.1 # fraction of signature
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
                 templates=None,
                 power_result_info=None,
                 test_recording=None,
                 test_sel_tbl=None,
                 apply_bandpass=False
                 ):
        '''
        Constructor
        '''
        self.log = LoggingService()
        
        self.species = species
        self.actions = actions if type(actions) == list else [actions]
        self.test_recording = test_recording
        self.test_sel_tbl = test_sel_tbl
        
        self._init_data_paths()

        # Cache of loaded audio for visualizations
        # maps fileName --> np_array
        self.audio_dict = {}
        # Map audioFile to Raven spectrogram df
        self.spectro_dict = {}
        
        experiment_root = os.path.abspath(os.path.join(self.experiment_dir, 
                                                         experiment_name))
        self.experiment = ExperimentManager(experiment_root)
        
        # Was a PowerResult computed, and stored as json either
        # in experiment, or elsewhere on the file system, materialize
        # that PowerResult instance:
        if power_result_info is not None:
            pwr_res = self.experiment.read(power_result_info, PowerResult)
        else:
            # See whether the experiment has PowerResult instances:
            pwr_res = self._latest_power_result(self.experiment)

        if templates is None:
            # Does the experiment have templates from a
            # prior quad_sig_calibration run?
            try:
                templates = self.experiment.read('signatures', QuadSigCalibrator)
            except FileNotFoundError:
                # The templates remains at None
                pass
        else:
            # Path to signatures json file stored by QuadSigCalibrator:
            templates_dict = QuadSigCalibrator.json_load(templates)
            template = templates_dict[species]

        self.template = template
        
        self.power_member = PowerMember(
            species_name=species, 
            spectral_template_info=template,
            the_slide_width_time=self.SLIDE_WIDTH,
            experiment=self.experiment,
            apply_bandpass=apply_bandpass
            )

        for action in self.actions:
            if action == Action.UNITTEST:
                return
            elif action == Action.GRID_SEARCH:
                grid_res = self.grid_search()
            elif action == Action.TEST:
                self.log.info("Running action 'test'...")
                pwr_res = self.run_test(self.test_sel_tbl,
                                        pwr_res=pwr_res,
                                        pwr_member=self.power_member,
                                        rec_path=self.test_recording,
                                        )
                self.log.info("Done running action 'test'.")
                self.log.info(f"Saving power result under 'pwr_res' to exp {self.experiment.root}")
                self.experiment.save('pwr_res', pwr_res)
            elif action == Action.VIZ_PROBS:
                ax = self.viz_probs(self.power_member,
                                    self.test_recording,
                                    self.test_sel_tbl,
                                    self.SIG_ID
                                    )
            else:
                raise ValueError(f"Unknown action: {action}")

        print("Done")

    #------------------------------------
    # run_test
    #-------------------
    
    def run_test(self,
                 sel_tbl_path,
                 pwr_res=None,
                 pwr_member=None,
                 rec_path=None
                 ):

        if pwr_res is None and pwr_member is None:
            raise ValueError("Either pwr_res or pwr_member plus rec_path must be provided")
            
        if pwr_res is None:
            pwr_res = pwr_member.compute_probabilities(rec_path) 
            
            # Add a Truth column to the result:
            pwr_res.add_truth(sel_tbl_path)
            
            pwr_res_fname = FileUtils.construct_filename(
                props_info = {'species' : pwr_res.species},
                suffix='.json',
                prefix='PwrRes',
                incl_date=True
                )
            self.experiment.save(pwr_res_fname, pwr_res)

        _axes = pwr_member.plot_pr_curve(pwr_res)
        
        threshold = 0.75
        
        scores = {}
        for sig_id in pwr_res.sig_ids():
            quantile_evaluator = PowerQuantileClassifier(sig_id=sig_id, 
                                                         threshold_quantile=threshold)
            quantile_evaluator.fit(pwr_res)
            score_name = f"score_sig_id{sig_id}_thres{threshold}" 
            scores[score_name] = quantile_evaluator.score(None, None, name=score_name)
        
        input("Press any key to quit: ")
        print('foo')
        #call_intervals = Utils.get_call_intervals(sel_tbl_path)

        #clf = PowerQuantileClassifier(sig_id, thres)
        #clf.fit(pwr_res)
        #score = clf.score(pwr_res.probabilities(sig_id), 
        #                  pwr_res.truths(sig_id),
        #                  name=f"({self.SIG_ID}/{self.THRESHOLD}/{self.SLIDE_WIDTH})"
        #                  )
        #return score

    #------------------------------------
    # grid_search
    #-------------------
    
    def grid_search(self):
        
        # pr_disp = sklearn.metrics.PrecisionRecallDisplay.from_estimator(
        #     clf,
        #     pwr_res.prob_df.probability,
        #     pwr_res.prob_df.Truth
        #     )
        # pr_disp.plot()
        from timeit import default_timer as timer

        start = timer()
        # 1hr:20min
        _grid_res = PowerQuantileClassifier.grid_search(
            self.power_member,
            audio_file=self.test_recording,
            selection_tbl_file=self.test_sel_tbl,
            #******quantile_thresholds=[0.80, 0.85, 0.90, 0.99],
            #******quantile_thresholds=[0.18, 0.19, 0.20, 0.80],
            quantile_thresholds=[0.21, 0.22, 0.3],
            slide_widths=[0.01, 0.02, 0.03],
            experiment=self.experiment
            )
        end = timer()
        self.log.info(f"Runtime: {str(timedelta(seconds=end - start))}")
        self.log.info(f"Grid result in {self.experiment.root}")
        
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
            pwr_res.add_truth(test_sel_tbl)
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
                              shading='auto')
        
        ax = mesh.axes

        call_intervals = Utils.get_call_intervals(test_sel_tbl)
        truth_rects_x = [interval['low_val']
                         for interval
                         in call_intervals]
        truth_rects_y = [0]*len(call_intervals)
        widths        = [interval['high_val'] - interval['low_val']
                         for interval
                         in call_intervals]
        heights       = [self.TRUTH_RECT_HEIGHTS]*len(call_intervals)

        fig = ax.figure
        fig.suptitle(f"{os.path.basename(test_recording)} ({power_member.species_name})")
        fig.show()
        
        for x,y,width,height in zip(truth_rects_x, truth_rects_y, widths, heights):
            ax.add_patch(Rectangle((x,y),
                                   width, height,
                                   facecolor=None,
                                   fill=False,
                                   edgecolor='black',
                                   hatch='*')
                                   )
        cursor = PowerInfoCursor(ax, pwr_res, truths, call_intervals, spectro)
        _motion_conn_id = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        _click_conn_id  = fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        if Action.UNITTEST in self.actions:
            return fig, ax, cursor
        input("Press ENTER to finish: ")

# --------------  Utilities -----------------

    #------------------------------------
    # _latest_power_result
    #-------------------
    
    def _latest_power_result(self, exp):
        '''
        Given an experiment check whether it contains
        PowerResult instances. If so, return the newest
        instance.
        
        :param exp: experiment to check
        :type exp: ExperimentManager
        :return: newest power result or None if none available
        :rtype {None | PowerResult}
        '''

        # Dict mapping timestamps to file names:
        time_files = {}
        jfiles = exp.listdir(PowerResult)
        for jfile in jfiles:
            # Jfile names are like PwrRes_2022-01-02T11_45_07.json
            # Get the parts:
            fparts = FileUtils.parse_filename(jfile)
            try:
                if fparts['prefix'] != 'PwrRes':
                    continue
            except KeyError:
                # Filename does not even have a 'prefix' part to it:
                continue
            # Replace the underscores w/ colons to get
            # correct iso formated times:
            timestamp = fparts['timestamp'].replace('_',':')
            time_files[datetime.fromisoformat(timestamp)] = jfile
        if len(time_files) == 0:
            return None
        newest_date = max(time_files.keys())
        exp_key = time_files[newest_date]
        pwr_res = exp.read(exp_key, PowerResult)
        return pwr_res

    #------------------------------------
    # _init_data_paths
    #-------------------

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
        self.sel_tbl_cmto_xc1 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        self.sel_rec_cmto_xc1 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        self.sel_tbl_cmto_xc2 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto2.selections.txt')
        self.sel_rec_cmto_xc2 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        self.sel_tbl_cmto_xc3 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto3.selections.txt')
        self.sel_rec_cmto_xc3 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')

# ---------------- Class PowerInfoCursor --------

class PowerInfoCursor(CrosshairCursor):
    
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, ax, pwr_res, truths, call_intervals, spectro):
        CrosshairCursor.__init__(self, ax)
        
        ax.texts[0].set_position((0.08, 0.4))
        self.pwr_res = pwr_res
        self.truths = truths
        self.call_intervals = call_intervals
        self.spectro = spectro
        
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
        
        is_call = Interval.binary_search_contains(self.call_intervals, x) > -1
        # For each signature, the current probability:
        probs_by_sig = []
        for sig_idx in range(self.num_sigs):
            probs_ser = self.pwr_res.probabilities(sig_idx+1)
            prob_times = probs_ser.index.values
            prob_idx = bisect_left(prob_times, x)
            try:
                probs_by_sig.append(probs_ser.iloc[prob_idx].round(2))
            except IndexError:
                # ignore
                continue
        
        # Find closest freq and time values in axes space
        # from the cursor, and get the spectrogram value there:
        freq_idx = bisect_left(list(reversed(self.spectro.index)), y)
        time_idx = bisect_left(list(self.spectro.columns), x)

        try:
            dbfs = round(self.spectro.iloc[freq_idx, time_idx], 1)
        except IndexError:
            #print(f"*********** Index error: freq_idx: {freq_idx}, time_idx: {time_idx}")
            dbfs = 0.0
        
        if len(probs_by_sig) > 0:
            best_sig_idx = np.argmax(probs_by_sig) if is_call else np.argmin(probs_by_sig)
            best_sig_id = best_sig_idx + 1 # sig IDs are 1-based
        
            txt = (f"time={x.round(2)}, freq={int(y)}\n"
                   f"is call: {is_call}\n"
                   f"power: {dbfs} dB FS \n"
                   f"probs by sig: {probs_by_sig}\n"
                   f"best sig: {best_sig_id} ({probs_by_sig[best_sig_idx]})\n"
                   f"mean_prob: {np.mean(probs_by_sig)}\n"
                   f"median_prob: {np.median(probs_by_sig)}\n"
                   f"min_prob: {np.min(probs_by_sig)}\n"
                   f"max_prob: {np.max(probs_by_sig)}"
                   )
    
            self.text.set_text(txt)
            self._update_display()

    #------------------------------------
    # _update_display
    #-------------------
    
    def _update_display(self):
        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.horizontal_line)
        self.ax.draw_artist(self.vertical_line)
        self.ax.draw_artist(self.text)
        self.ax.figure.canvas.blit(self.ax.bbox)

    #------------------------------------
    # onclick
    #-------------------
    
    def onclick(self, event):
        '''
        Left/right clicks: increase/decrease font size
        Middle click: switch between black and white
        
        :param event:
        :type event:
        '''
        if event.button == 2:
            cur_color = self.text.get_color()
            if cur_color == 'black':
                self.text.set_color('white')
            else:
                self.text.set_color('black')
        elif event.button == 1:
            cur_fontsize = self.text.get_fontsize()
            self.text.set_fontsize(cur_fontsize + 2)
            # Position of lower left of text block in
            # axes coordinates (0,0 is lower left, 1,1 is upper right):
            #******cur_txt_x, cur_txt_y = self.text.get_position()
            # Move txt down a bit to accommodate the larger font:
            print(f"MB1: x: {event.x}; y: {event.y}, xdata: {event.xdata}, ydata: {event.ydata}")
            #*****self.text.set_position((cur_txt_x, cur_txt_y + 0.4))
            #*****self.text.set_position((event.x, event.y))
            self.text.set_position((event.x, event.y))
            self.text.set_transform(self.ax.transAxes)
            
        elif event.button == 3:
            cur_fontsize = self.text.get_fontsize()
            self.text.set_fontsize(cur_fontsize - 2)
            #****cur_txt_x, cur_txt_y = self.text.get_position()
            # Move txt up a bit b/c txt now smaller:
            print(f"MB3: x: {event.x}; y: {event.y}, xdata: {event.xdata}, ydata: {event.ydata}")
            #*****self.text.set_position((cur_txt_x, cur_txt_y - 0.4))
            self.text.set_position((event.xdata, event.ydata))
            self.text.set_transform(self.ax.transAxes)

        if event.button in [1,2,3]:
            self._update_display()

    #------------------------------------
    # _choose_text_color
    #-------------------

    # Not working; replaced with middle-click to switch
    # between left and right:
        
    # def _choose_text_color(self, spectro, txt_area, color_map_name='jet'):
    #     '''
    #     Given a spectrogram and a dict: {'y', 'x', 'width', 'height')
    #     that define a rectangle, compute the average color 
    #     in the rectangle, and retun 'black' or 'white' 
    #     for the color to use for text on top of the rectangle
    #     when the spectrogram is rendered.
    #
    #     Units are indices into the spectrogram index and
    #     columns(, which will correspond to labels on the
    #     axis scale ticks of a pcolormesh). Thus: width is time,
    #     and height is frequencies.
    #
    #     :param spectro: dataframe of values that will
    #         be shown in pcolormesh
    #     :type spectro: pd.DataFrame
    #     :param txt_area: a rectangle over the mesh as
    #         a dict with keys ['x','y','width','height']
    #     :type txt_area: (int, int,int, int)
    #     :returns {'black' | 'white'}
    #     :rtype str
    #     '''
    #
    #     yx = (txt_area['y'], txt_area['x'])
    #     rect_df = Utils.df_extract_rect(spectro,
    #                                     yx=yx, 
    #                                     height=txt_area['height'], 
    #                                     width=txt_area['width']) 
    #     rect_df_abs = rect_df.abs()
    #     rect_df_normed = rect_df_abs / rect_df_abs.max().max()
    #     mean_rect_normed = rect_df_normed.mean().mean() 
    #     cmap = matplotlib.cm.get_cmap(color_map_name)
    #     rgb_fractions = pd.Series(cmap(mean_rect_normed), index=['R', 'G', 'B', 'A'])
    #     rgb_255 = rgb_fractions * 255
    #     txt_color = 'white' if 1 - (rgb_255['R'] * 0.299 + \
    #                                 rgb_255['G'] * 0.587 + \
    #                                 rgb_255['B'] * 0.114) / 255 < 0.5 \
    #                         else 'black'
    #

    #    return txt_color

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Evaluate or calibrate the power signature approach"
                                     )

    parser.add_argument('--templates',
                        help='path to already computed templates',
                        default=None)
    # parser.add_argument('--sig_rec',
    #                     help='path to recording from which signatures are to be computed',
    #                     default=None)
    # parser.add_argument('--sig_sel',
    #                     help='path to selection table for signature',
    #                     default=None)
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
    parser.add_argument('-b', '--bandpass',
                        action='store_true',
                        help='apply bandpass filters to sigs and inferenced audio with the freq range of signatures default: False',
                        default=False)
    parser.add_argument('actions',
                        choices=['test', 'gridSearch', 'viz_probs'],
                        nargs='+',
                        help='Repeatable: actions to perform')

    args = parser.parse_args()

    cur_dir = os.path.dirname(__file__)
    # Check file existence:

    #if args.templates is None:
    #    default_templates_path = os.path.join(cur_dir, 'species_calibration_data/signatures.json')

    if args.templates is not None and not os.path.exists(args.templates):
        raise FileNotFoundError(f"Templates file not found ({args.templates})")

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
                   args.templates,
                   test_recording=args.test_rec,
                   test_sel_tbl=args.test_sel,
                   apply_bandpass=args.bandpass
                   )
    print('foo')