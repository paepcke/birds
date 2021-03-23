Currently in this <birds-root>/src/birdsong/tests/data_other
directory:

- inference_data:
     60 spectrograms from ten species.
     None of these spectros was used in
     training models in run_models.

     The files are from
     quintus:/home/data/birds/testset_for_taken_out_recombined_data/

- run_models:
     Dir:
     models_2021-03-23T08_21_15_net_resnet18_ini_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_gray_True_classes_10

     Inside: 20 models (epochs 3 to 22) from training
                quintus:/home/data/birds/taken_out_recombined_data

     The files under inference_data were removed from both training
     and validation before training.
     
