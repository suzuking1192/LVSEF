# Installation

## install dependencies
conda env create -f environment.yml

## install Retro model
Download and unzip the files from the link below, and put all the folders (dataset/, one_step_model/ and saved_models/) under the retro_star directory.

https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0

# Train
For Acrylates,

python main.py \
        --max_epoches 20 \
        --without_retro \
        --training_data=./datasets/acrylates.txt \
        --grammar_training \
        --grammar_training_round 5 \
        --grammar_training_round_sample 100 \
        --grammar_lr 0.01 \
        --grammar_explore_rate 0.005 \
        --grammar_reconstruction_addition 0.4 \
        --fragment_ranking_pred \
        --fragment_ranking_pred_gammar 0.95 \
        --random_warm_up 20 \
        --task_name parameter_analysis_ours_acry_lr_0.01_expl_0.005_recon_0.4

For Polymer,

python main.py \
        --max_epoches 20 \
        --sa_score \
        --training_data=./datasets/polymers_117.txt \
        --grammar_training \
        --grammar_training_round 5 \
        --grammar_training_round_sample 100 \
        --grammar_lr 0.05 \
        --grammar_explore_rate 0.001 \
        --grammar_reconstruction_addition 0.5 \
        --sa_score \
        --sa_thres_train_data 3.5 \
        --fragment_ranking_pred \
        --fragment_ranking_pred_gammar 0.95 \
        --random_warm_up 20 \
        --task_name ours_poly_lr_0.05_expl_0.001_recon_0.5_batch_size_50_sa_thres_3.5_q_table_1000 \
        --batch_training \
        --batch_size 50 \
        --remove_q_table_thres 1000

# Evaluation
For Acrylates,
python evaluate.py --training_data ./datasets/acrylates.txt \
                --expr_name acrylates \
                --model_folder  log/log-num_generated_samples100-20240305-095612_grammar_training_True_grammar_lr_0.01_grammar_explore_rate0.005_acrylatess_lr_0.01_explore_0.005_recon_0.3\
                --num_generated_samples 1000 \
                --grammar_training \
                --early_stopping 100 \
                --add_top_k 110 \
                --topk_from_scratch \
                --top_k_selection 20 \
                --random_iter_start 5 \
                --final_model \
                --a_ending_factor 0.4 \
                --task_name generation_ablation_acr_top_110_scratch_our_combination_final_model_a_0.4_save_smi_top_k_selection_20_random_iter_start_5 \
                --save_grammar_sample_smi

For Polymer,
python evaluate.py --training_data ./datasets/polymers_117.txt \
                --expr_name polymer \
                --model_folder  log/log-num_generated_samples100-20240326-181216_grammar_training_True_grammar_lr_0.05_grammar_explore_rate0.001_ours_poly_lr_0.05_expl_0.001_recon_0.5_batch_size_50_sa_thres_3.5_q_table_1000\
                --num_generated_samples 1000 \
                --grammar_training \
                --early_stopping 100 \
                --add_top_k 90 \
                --topk_from_scratch \
                --add_top_k_starting_ending 20 \
                --final_model \
                --a_ending_factor 0.6 \
                --task_name ours_poly_top90_startingrule_20_sa_thres_3.5_enda_0.6 \
                --save_grammar_sample_smi \
                --save_sa_score