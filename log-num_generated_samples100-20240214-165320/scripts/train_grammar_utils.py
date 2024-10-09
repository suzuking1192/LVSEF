import pickle

import numpy as np

from grammar_generation import *
from private import *


def evaluate_by_trained_grammar(grammar, args, grammar_model, metrics=["diversity", "syn"]):
    # Metric evalution for the given gramamr
    div = InternalDiversity()
    eval_metrics = {}
    generated_samples = []
    generated_samples_canonical_sml = []
    iter_num_list = []
    idx = 0
    no_newly_generated_iter = 0
    print("Start grammar evaluation...")
    while True:
        print("Generating sample {}/{}".format(idx, args.num_generated_samples))
        mol, iter_num = grammar_model.produce(grammar)
        if mol is None:
            no_newly_generated_iter += 1
            continue
        can_sml_mol = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        if can_sml_mol not in generated_samples_canonical_sml:
            generated_samples.append(mol)
            generated_samples_canonical_sml.append(can_sml_mol)
            iter_num_list.append(iter_num)
            idx += 1
            no_newly_generated_iter = 0
        else:
            no_newly_generated_iter += 1
        if idx >= args.num_generated_samples or no_newly_generated_iter > 10:
            break

    for _metric in metrics:
        assert _metric in ["diversity", "num_rules", "num_samples", "syn"]
        if _metric == "diversity":
            diversity = div.get_diversity(generated_samples)
            eval_metrics[_metric] = diversity
        elif _metric == "num_rules":
            eval_metrics[_metric] = grammar.num_prod_rule
        elif _metric == "num_samples":
            eval_metrics[_metric] = idx
        elif _metric == "syn":
            eval_metrics[_metric] = retro_sender(generated_samples, args)
        else:
            raise NotImplementedError
    return eval_metrics


class markov_grammar_model:

    def __init__(self, grammar, lr) -> None:

        self.lr = lr
        self.q_table = np.zeros((len(grammar), len(grammar)))
        self.all_grammar = grammar

    def check_grammar_is_in_q_table(self, grammar_rule):
        pass

    def add_grammar_rule(self, grammar_rule):
        pass

    def adjust_q_table_columns(self, grammar):
        for grammar_rule in grammar:
            if self.check_grammar_is_in_q_table(grammar_rule):
                pass
            else:
                self.add_grammar_rule(grammar_rule)

    def train_markov_grammar_by_reconstruction(self, grammar):
        pass

    def train_markov_grammar(self, grammar):
        pass

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def produce(self, grammar):
        return mol, iter_num
