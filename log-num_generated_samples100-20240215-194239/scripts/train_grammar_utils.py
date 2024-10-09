import copy
import fcntl
import pickle
import time

import numpy as np

from grammar_generation import *
from private import *
from retro_star_listener import *


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
        self.q_table = np.zeros((len(grammar.prod_rule_list), len(grammar.prod_rule_list)))
        self.grammar_index = grammar.prod_rule_list

        self.reconstruction_addition = 0.5

        self.tmp_current_grammar_idx = None

    def check_grammar_is_in_q_table(self, grammar_rule):
        for grammar_rule_in_q_table in self.grammar_index:
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
        for grammar_rule in grammar.prod_rule_list:
            if self.check_grammar_is_in_q_table(grammar_rule):
                pass
            else:
                self.add_grammar_rule(grammar_rule)

    def get_grammar_rule_idx(self, grammar_rule):
        counter = 0
        for grammar_rule_in_q_table in self.grammar_index:
            if grammar_rule_in_q_table.is_same(grammar_rule):
                return counter
            counter += 1

    def update_q_table_by_rule(self, condition_grammar_rule, action_grammar_rule):
        condition_rule_idx = self.get_grammar_rule_idx(condition_grammar_rule)
        action_rule_idx = self.get_grammar_rule_idx(action_grammar_rule)

        self.q_table[condition_rule_idx][action_rule_idx] += self.reconstruction_addition

    def update_q_table_by_reconstruction(self, input_g):
        print("start calculating reconstruction loss....")
        grammar_rule_list = copy.deepcopy(input_g.rule_list)

        if len(grammar_rule_list) == 2:
            self.update_q_table_by_rule(grammar_rule_list[1], grammar_rule_list[0])
        elif len(grammar_rule_list) == 3:
            self.update_q_table_by_rule(grammar_rule_list[2], grammar_rule_list[1])
            self.update_q_table_by_rule(grammar_rule_list[1], grammar_rule_list[0])
        else:
            start_rule_idx = len(grammar_rule_list) - 1
            end_rule_idx = 0
            rule_order_idx = [start_rule_idx]
            choice_rule_idx = [a for a in range(len(grammar_rule_list) - 1) if a > 0]

            hypergraph = Hypergraph()
            first_rule = grammar_rule_list[start_rule_idx]
            hg_cand, _, avail = first_rule.graph_rule_applied_to(hypergraph)
            hypergraph = deepcopy(hg_cand)
            while len(choice_rule_idx) > 1:
                find_next_motif = False
                next_motif_idx_option = choice_rule_idx[-1]
                while find_next_motif == False:

                    next_rule = grammar_rule_list[next_motif_idx_option]
                    hg_cand, _, avail = next_rule.graph_rule_applied_to(hypergraph)
                    if hg_cand.is_subhg(input_g.hypergraph) == True:
                        find_next_motif = True
                        rule_order_idx.append(next_motif_idx_option)
                        hypergraph = deepcopy(hg_cand)

                        choice_rule_idx.remove(next_motif_idx_option)
                    else:
                        next_motif_idx_option -= 1
            # add last two grammars
            print(choice_rule_idx)

            rule_order_idx.append(choice_rule_idx[0])
            rule_order_idx.append(end_rule_idx)

            counter = 0
            for rule_idx in rule_order_idx:
                if rule_idx != end_rule_idx:

                    self.update_q_table_by_rule(grammar_rule_list[rule_idx], grammar_rule_list[rule_order_idx[counter + 1]])
                counter += 1

    def train_markov_grammar_by_reconstruction(self, input_graphs_dict):
        for i, (key, input_g) in enumerate(input_graphs_dict.items()):
            self.update_q_table_by_reconstruction(input_g)

    def train_markov_grammar(self, grammar, num_group_eval=10):
        chosen_grammar_idx_list = []
        syn = Synthesisability()
        div = InternalDiversity()
        syn_status = []
        mol_list = []
        for i in range(num_group_eval):
            mol, _, chosen_grammar_idx = self.produce(grammar)
            chosen_grammar_idx_list.append(chosen_grammar_idx)
            mol_list.append(mol)
            try:
                result = syn.planner.plan(Chem.MolToSmiles(mol))
            except:
                result = None
            result = -1 if result is None else 1
            syn_status.append(result)
        diversity = div.get_diversity(mol_list)

        for group_i in range(num_group_eval):
            group_reward = diversity + syn_status[group_i]
            chosen_grammar_idx = chosen_grammar_idx_list[group_i]
            for i in range(len(chosen_grammar_idx) - 1):
                condition_grammar_idx = chosen_grammar_idx[i]
                action_grammar_idx = chosen_grammar_idx[i + 1]

                condition_grammar_idx_q_table = self.tmp_current_grammar_idx[condition_grammar_idx]
                action_grammar_idx_q_table = self.tmp_current_grammar_idx[action_grammar_idx]

                self.q_table[condition_grammar_idx_q_table][action_grammar_idx_q_table] += group_reward

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def update_tmp_current_grammar_idx(self, grammar):
        tmp_current_grammar_idx = []
        for grammar_rule in grammar.prod_rule_list:
            grammar_rule_idx = self.get_grammar_rule_idx(self, grammar_rule)
            tmp_current_grammar_idx.append(grammar_rule_idx)
        self.tmp_current_grammar_idx = tmp_current_grammar_idx

    def produce(self, grammar):
        def sample(l, prob=None):
            if prob is None:
                prob = [1 / len(l)] * len(l)
            idx = np.random.choice(range(len(l)), 1, p=prob)[0]
            return l[idx], idx

        def prob_schedule(_iter, selected_idx, last_rule_idx):
            prob_list = []
            # prob = exp(a * t * x), x = {0, 1}
            # calculate probability based on Q_table
            condition_grammar_idx_q_table = self.tmp_current_grammar_idx[last_rule_idx]
            a = 0.5
            for rule_i, rule in enumerate(grammar.prod_rule_list):
                grammar_idx_q_table = self.tmp_current_grammar_idx[rule_i]
                q_table_prob = self.q_table[condition_grammar_idx_q_table][grammar_idx_q_table]
                x = rule.is_ending
                if rule.is_start_rule:
                    prob_list.append(0)
                else:
                    prob_list.append(np.exp(a * _iter * x) * q_table_prob)
            prob_list = np.array(prob_list)[selected_idx]
            prob_list = prob_list / np.sum(prob_list)
            return prob_list

        chosen_grammar_idx = []
        hypergraph = Hypergraph()
        starting_rules = [(rule_i, rule) for rule_i, rule in enumerate(grammar.prod_rule_list) if rule.is_start_rule]
        iter = 0
        while True:
            if iter == 0:
                _, idx = sample(starting_rules)
                selected_rule_idx, selected_rule = starting_rules[idx]
                hg_cand, _, avail = selected_rule.graph_rule_applied_to(hypergraph)
                hypergraph = deepcopy(hg_cand)
                last_rule_idx = selected_rule_idx
                chosen_grammar_idx.append(selected_rule_idx)
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
                prob_list = prob_schedule(iter, candidate_rule_idx, last_rule_idx)
                hypergraph, idx = sample(candidate_hg, prob_list)
                selected_rule = candidate_rule_idx[idx]
                last_rule_idx = idx
                chosen_grammar_idx.append(last_rule_idx)
            iter += 1
        try:
            mol = hg_to_mol(hypergraph)
            print(Chem.MolToSmiles(mol))
        except:
            return None, iter, chosen_grammar_idx

        return mol, iter, chosen_grammar_idx
