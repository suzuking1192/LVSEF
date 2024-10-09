python main.py --max_epoches 2 \
                --without_retro \
                --training_data=./datasets/chain_extenders.txt \
                --grammar_training \
                --grammar_training_round 2 \
                --grammar_training_round_sample 10 \
                --fragment_ranking_pred \
                --random_warm_up 1