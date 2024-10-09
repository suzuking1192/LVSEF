import fcntl
import pickle
import time

import numpy as np

from grammar_generation import *
from private import *
from retro_star_listener import lock


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


def retro_sender(generated_samples, args):
    # File communication to obtain retro-synthesis rate
    with open(args.receiver_file, "w") as fw:
        fw.write("")
    while True:
        with open(args.sender_file, "r") as fr:
            editable = lock(fr)
            if editable:
                with open(args.sender_file, "w") as fw:
                    for sample in generated_samples:
                        fw.write("{}\n".format(Chem.MolToSmiles(sample)))
                break
            fcntl.flock(fr, fcntl.LOCK_UN)
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    while True:
        with open(args.receiver_file, "r") as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    assert len(generated_samples) == len(syn_status)
    return np.mean([int(eval(s[1])) for s in syn_status])


def check_grammar_rule_is_same(grammar_rule_1, grammar_rule_2):
    pass


class markov_grammar_model:

    def __init__(self, grammar, lr) -> None:

        self.lr = lr
        self.q_table = np.zeros((len(grammar), len(grammar)))
        self.grammar_index = grammar

    def check_grammar_is_in_q_table(self, grammar_rule):
        for grammar_rule_in_q_table in self.all_grammar:
            if grammar_rule_in_q_table.is_same(grammar_rule):
                return True

        return False

    def add_grammar_rule(self, grammar_rule):
        self.grammar_index.append(grammar_rule)
        # Add one row of zeros at the bottom
        self.q_table = np.vstack([self.q_table, np.zeros((1, self.q_table.shape[1]))])

        # Add one column of zeros on the right
        self.q_table = np.hstack([self.q_table, np.zeros((self.q_table.shape[0], 1))])

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
        def sample(l, prob=None):
            if prob is None:
                prob = [1 / len(l)] * len(l)
            idx = np.random.choice(range(len(l)), 1, p=prob)[0]
            return l[idx], idx

        def prob_schedule(_iter, selected_idx):
            prob_list = []
            # prob = exp(a * t * x), x = {0, 1}
            # calculate probability based on Q_table
            return prob_list

        hypergraph = Hypergraph()
        starting_rules = [(rule_i, rule) for rule_i, rule in enumerate(grammar.prod_rule_list) if rule.is_start_rule]
        iter = 0
        while True:
            if iter == 0:
                _, idx = sample(starting_rules)
                selected_rule_idx, selected_rule = starting_rules[idx]
                hg_cand, _, avail = selected_rule.graph_rule_applied_to(hypergraph)
                hypergraph = deepcopy(hg_cand)
            else:
                candidate_rule = []
                candidate_rule_idx = []
                candidate_hg = []
                for rule_i, rule in enumerate(grammar.prod_rule_list):
                    hg_prev = deepcopy(hypergraph)
                    hg_cand, _, avail = rule.graph_rule_applied_to(hypergraph)
                    if avail:
                        candidate_rule.append(rule)
                        candidate_rule_idx.append(rule_i)
                        candidate_hg.append(hg_cand)
                if (all([rl.is_start_rule for rl in candidate_rule]) and iter > 0) or iter > 30:
                    break
                prob_list = prob_schedule(iter, candidate_rule_idx)
                hypergraph, idx = sample(candidate_hg, prob_list)
                selected_rule = candidate_rule_idx[idx]
            iter += 1
        try:
            mol = hg_to_mol(hypergraph)
            print(Chem.MolToSmiles(mol))
        except:
            return None, iter

        return mol, iter
