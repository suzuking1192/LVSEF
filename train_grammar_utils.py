import numpy as np
from private import *
from grammar_generation import *
import pickle
import fcntl
# from retro_star_listener import *
import time
import copy


def without_retro(generated_samples, args):
    syn = Synthesisability()
    syn_status = []
    for sample_smi in generated_samples:
        print("predict synthesizability")
        try:

            result = syn.planner.plan(Chem.MolToSmiles(sample_smi))
        except:
            result = None
        result = False if result is None else True
        syn_status.append(result)
    return sum(syn_status) / len(syn_status)


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
        mol, iter_num, _ = grammar_model.produce(grammar)
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
            if args.sa_score:
                eval_metrics[_metric] = 1 - calculate_average_synth_score(generated_samples) / 10

            elif args.without_retro:
                eval_metrics[_metric] = without_retro(generated_samples, args)
            else:
                eval_metrics[_metric] = retro_sender(generated_samples, args)
        else:
            raise NotImplementedError
    return eval_metrics


from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def calculate_average_synth_score(mol_list):
    total_score = 0
    valid_count = 0

    for mol in mol_list:

        if mol is not None:
            try:
                score = sascorer.calculateScore(mol)
                total_score += score
                valid_count += 1
            except:
                total_score += 10
                valid_count += 1

    if valid_count == 0:
        return None
    else:
        return total_score / valid_count


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


class markov_grammar_model:

    def __init__(self, grammar, lr, explore_rate, reconstruction_addition, select_frag=False, frag_list=None) -> None:

        self.lr = lr
        self.explore_rate = explore_rate
        self.q_table = np.ones((len(grammar.prod_rule_list), len(grammar.prod_rule_list))) * self.explore_rate
        self.grammar_index = grammar.prod_rule_list

        self.reconstruction_addition = reconstruction_addition

        self.tmp_current_grammar_idx = None

        self.select_fragments = select_frag
        self.frag_list = frag_list

    def get_q_table(self):
        return self.q_table

    def check_grammar_is_in_q_table(self, grammar_rule):
        for grammar_rule_in_q_table in self.grammar_index:
            if grammar_rule_in_q_table.is_same(grammar_rule)[0]:
                return True

        return False

    def add_grammar_rule(self, grammar_rule):
        self.grammar_index.append(grammar_rule)
        # Add one row of zeros at the bottom
        self.q_table = np.vstack([self.q_table, self.explore_rate * np.ones((1, self.q_table.shape[1]))])

        # Add one column of zeros on the right
        self.q_table = np.hstack([self.q_table, self.explore_rate * np.ones((self.q_table.shape[0], 1))])

    def adjust_q_table_columns(self, grammar):
        for grammar_rule in grammar.prod_rule_list:
            if self.check_grammar_is_in_q_table(grammar_rule):
                pass
            else:
                self.add_grammar_rule(grammar_rule)

    def get_grammar_rule_idx(self, grammar_rule):
        counter = 0
        for grammar_rule_in_q_table in self.grammar_index:

            if grammar_rule_in_q_table.is_same(grammar_rule)[0]:
                return counter
            counter += 1

    def update_q_table_by_rule(self, condition_grammar_rule, action_grammar_rule):
        condition_rule_idx = self.get_grammar_rule_idx(condition_grammar_rule)
        action_rule_idx = self.get_grammar_rule_idx(action_grammar_rule)
        try:
            self.q_table[condition_rule_idx][action_rule_idx] += self.reconstruction_addition
        except:
            # FIX LATER
            print("FIX LATER not able to find grammar idx")

    def update_q_table_by_reconstruction(self, input_g):
        print("start calculating reconstruction loss....")
        grammar_rule_list = copy.deepcopy(input_g.rule_list)

        if len(grammar_rule_list) == 2:
            self.update_q_table_by_rule(grammar_rule_list[1], grammar_rule_list[0])
        elif len(grammar_rule_list) == 3:
            self.update_q_table_by_rule(grammar_rule_list[2], grammar_rule_list[1])
            self.update_q_table_by_rule(grammar_rule_list[1], grammar_rule_list[0])
        elif len(grammar_rule_list) < 2:
            pass
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

                        if next_motif_idx_option < 0:
                            print("select from past grammars")
                        else:
                            choice_rule_idx.remove(next_motif_idx_option)
                    else:
                        # when we need to change the order of grammar

                        next_option_idx_list = []
                        for rule_i in choice_rule_idx:
                            next_rule = grammar_rule_list[rule_i]
                            hg_cand, _, avail = next_rule.graph_rule_applied_to(hypergraph)
                            if hg_cand.is_subhg(input_g.hypergraph) == True:
                                next_option_idx_list.append(rule_i)

                        if len(next_option_idx_list) == 1:
                            rule_i = next_option_idx_list[0]
                            next_rule = grammar_rule_list[rule_i]
                            hg_cand, _, avail = next_rule.graph_rule_applied_to(hypergraph)
                            find_next_motif = True
                            rule_order_idx.append(rule_i)
                            hypergraph = deepcopy(hg_cand)
                            choice_rule_idx.remove(rule_i)
                        elif len(next_option_idx_list) == 0:
                            find_next_motif = True
                            choice_rule_idx = []
                        else:
                            print("FIX LATER to get precise order of reconstruction")
                            # FIX LATER to get precise order of reconstruction
                            find_next_motif = True
                            rule_i = next_option_idx_list[0]
                            choice_rule_idx = [rule_i]
            # add last two grammars

            if len(choice_rule_idx) == 0:
                print("FIX LATER to get precise order of reconstruction")
            else:
                rule_order_idx.append(choice_rule_idx[0])
            rule_order_idx.append(end_rule_idx)

            counter = 0
            for rule_idx in rule_order_idx:
                if rule_idx != end_rule_idx:

                    self.update_q_table_by_rule(grammar_rule_list[rule_idx], grammar_rule_list[rule_order_idx[counter + 1]])
                counter += 1

    def train_markov_grammar_by_reconstruction(self, input_graphs_dict, low_sa_idxs=[]):
        for i, (key, input_g) in enumerate(input_graphs_dict.items()):
            if len(low_sa_idxs) > 0:
                if i in low_sa_idxs:
                    self.update_q_table_by_reconstruction(input_g)
            else:
                self.update_q_table_by_reconstruction(input_g)

    def train_markov_grammar(self, grammar, args, num_group_eval=10):
        chosen_grammar_idx_list = []
        if args.sa_score:
            pass
        else:
            syn = Synthesisability()
        div = InternalDiversity()
        syn_status = []
        mol_list = []
        for i in range(num_group_eval):
            mol, _, chosen_grammar_idx = self.produce(grammar)
            chosen_grammar_idx_list.append(chosen_grammar_idx)
            mol_list.append(mol)
            if args.sa_score:
                try:
                    result = (0.5 - sascorer.calculateScore(mol) / 10) * 2
                except:
                    result = -1
            else:
                try:
                    result = syn.planner.plan(Chem.MolToSmiles(mol))
                except:
                    result = None
                result = -1 if result is None else 1
            syn_status.append(result)
        try:
            diversity = div.get_diversity(mol_list)
        except:
            # remove invalid mols
            valid_mols = []
            for mol in mol_list:
                if mol is not None and Chem.MolToSmiles(mol) != "":
                    valid_mols.append(mol)
            try:
                diversity = div.get_diversity(valid_mols)
            except:
                # FIX LATER
                diversity = 0.5

        for group_i in range(num_group_eval):
            group_reward = diversity + syn_status[group_i]
            chosen_grammar_idx = chosen_grammar_idx_list[group_i]
            for i in range(len(chosen_grammar_idx) - 1):
                condition_grammar_idx = chosen_grammar_idx[i]
                action_grammar_idx = chosen_grammar_idx[i + 1]

                condition_grammar_idx_q_table = self.tmp_current_grammar_idx[condition_grammar_idx]
                action_grammar_idx_q_table = self.tmp_current_grammar_idx[action_grammar_idx]

                self.q_table[condition_grammar_idx_q_table][action_grammar_idx_q_table] += group_reward
                if self.q_table[condition_grammar_idx_q_table][action_grammar_idx_q_table] < 0:
                    self.q_table[condition_grammar_idx_q_table][action_grammar_idx_q_table] = 0
        print(f"diversity: {diversity}, synthesizability: {sum(syn_status) / len(syn_status)}")

        return diversity, sum(syn_status) / len(syn_status)

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def update_tmp_current_grammar_idx(self, grammar):
        tmp_current_grammar_idx = []
        for grammar_rule in grammar.prod_rule_list:
            grammar_rule_idx = self.get_grammar_rule_idx(grammar_rule)
            tmp_current_grammar_idx.append(grammar_rule_idx)
        self.tmp_current_grammar_idx = tmp_current_grammar_idx
        print("tmp_current_grammar_idx:", tmp_current_grammar_idx)

    def produce(self, grammar, uniform_smoothing_e=0, a=0.5, top_k_selection=0, random_iter_start=15):
        def sample(l, prob=None):
            if prob is None:
                prob = [1 / len(l)] * len(l)
            try:
                idx = np.random.choice(range(len(l)), 1, p=prob)[0]
            except:
                # FIX LATER for NaN error
                print("probability NaN error")
                try:
                    new_prob = [0 if np.isnan(item) else item for item in prob]
                    idx = np.random.choice(range(len(l)), 1, p=new_prob)[0]
                except:
                    print("probability NaN error (double error)")
                    prob = [1 / len(l)] * len(l)
                    idx = np.random.choice(range(len(l)), 1, p=prob)[0]
            return l[idx], idx

        def prob_schedule(
            _iter, selected_idx, last_rule_idx, uniform_smoothing_e=0, a=0.5, top_k_selection=0, random_iter_start=15
        ):
            prob_list = []
            # prob = exp(a * t * x), x = {0, 1}
            # calculate probability based on Q_table
            condition_grammar_idx_q_table = self.tmp_current_grammar_idx[last_rule_idx]
            # a = 0.5
            for rule_i, rule in enumerate(grammar.prod_rule_list):
                grammar_idx_q_table = self.tmp_current_grammar_idx[rule_i]

                try:
                    q_table_prob = copy.deepcopy(self.q_table[condition_grammar_idx_q_table][grammar_idx_q_table])
                except:
                    # FIX LATER None index of grammar
                    q_table_prob = 0

                x = rule.is_ending
                if rule.is_start_rule:
                    prob_list.append(0)
                else:
                    prob = np.exp(a * _iter * x) * q_table_prob

                    try:
                        if prob < 0:
                            prob = 0
                    except:
                        # FIX LATER grammar_idx_q_table is None
                        prob = 0
                    prob_list.append(prob)
            prob_list = np.array(prob_list)[selected_idx]
            prob_list = prob_list / np.sum(prob_list)

            if uniform_smoothing_e > 0:
                try:
                    num_events = len(prob_list)
                    smoothed_probabilities = [
                        (p + uniform_smoothing_e) / (1 + num_events * uniform_smoothing_e) for p in prob_list
                    ]
                    prob_list = smoothed_probabilities
                except:
                    pass

            if top_k_selection > 0 and _iter < random_iter_start:

                def equalize_top_k_probabilities(probabilities, k):
                    # Sort the probabilities and get the indices of the top k probabilities
                    sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:k]

                    # Initialize the equalized probabilities list with zeros
                    equalized_probabilities = [0] * len(probabilities)

                    # Set the top k probabilities to 1/k
                    for index in sorted_indices:
                        equalized_probabilities[index] = 1 / k

                    return equalized_probabilities

                prob_list = equalize_top_k_probabilities(prob_list, top_k_selection)

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

                    if self.select_fragments:
                        hg_cand, _, avail = rule.graph_rule_applied_to_frag_selection(hypergraph)
                    else:
                        hg_cand, _, avail = rule.graph_rule_applied_to(hypergraph)
                    if avail:
                        candidate_rule.append(rule)
                        candidate_rule_idx.append(rule_i)
                        candidate_hg.append(hg_cand)
                if (all([rl.is_start_rule for rl in candidate_rule]) and iter > 0) or iter > 30:
                    break
                prob_list = prob_schedule(
                    iter, candidate_rule_idx, last_rule_idx, uniform_smoothing_e, a, top_k_selection, random_iter_start
                )

                hypergraph, idx = sample(candidate_hg, prob_list)

                selected_rule = candidate_rule_idx[idx]
                last_rule_idx = idx
                chosen_grammar_idx.append(last_rule_idx)
            iter += 1
        try:
            mol = hg_to_mol(hypergraph)
            print(Chem.MolToSmiles(mol))
        except:
            print("error in generating samples")
            return None, iter, chosen_grammar_idx

        return mol, iter, chosen_grammar_idx

    def reduce_q_table_columns(self, num_col):
        q_score_list = []
        grammar_list_q_table = copy.deepcopy(self.grammar_index)

        for g_idx in range(len(grammar_list_q_table)):

            q_score = 0
            for t_idx in range(len(grammar_list_q_table)):

                q_score += self.q_table[t_idx][g_idx]
                q_score += self.q_table[g_idx][t_idx]

            q_score_list.append(q_score)

        top_k_indices = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)[:num_col]

        self.grammar_index = [value for index, value in enumerate(self.grammar_index) if index in top_k_indices]
        self.q_table = self.q_table[np.ix_(top_k_indices, top_k_indices)]
