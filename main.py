from private.hypergraph import mol_to_hg,mol_to_hg_add_dummy
from rdkit import Chem
from copy import deepcopy
import numpy as np
from private import *
from grammar_generation import *
from agent import Agent
import torch.optim as optim
import torch.multiprocessing as mp
import logging
import torch
import math
import os
import time
import pprint
import pickle
import argparse
import fcntl
import copy
from retro_star_listener import *
from train_grammar_utils import *

def evaluate(grammar, args, metrics=['diversity', 'syn']):
    # Metric evalution for the given gramamr
    div = InternalDiversity()
    eval_metrics = {}
    generated_samples = []
    generated_samples_canonical_sml = []
    iter_num_list = []
    idx = 0
    no_newly_generated_iter = 0
    print("Start grammar evaluation...")
    while(True):
        print("Generating sample {}/{}".format(idx, args.num_generated_samples))
        mol, iter_num = random_produce(grammar)
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
        assert _metric in ['diversity', 'num_rules', 'num_samples', 'syn']
        if _metric == 'diversity':
            diversity = div.get_diversity(generated_samples)
            eval_metrics[_metric] = diversity
        elif _metric == 'num_rules':
            eval_metrics[_metric] = grammar.num_prod_rule
        elif _metric == 'num_samples':
            eval_metrics[_metric] = idx
        elif _metric == 'syn':
            if args.sa_score:
                eval_metrics[_metric] = 1 - calculate_average_synth_score(generated_samples)/10
            elif args.without_retro:
                eval_metrics[_metric] = without_retro(generated_samples, args)
            else:
                eval_metrics[_metric] = retro_sender(generated_samples, args)
        else:
            raise NotImplementedError
    return eval_metrics


def retro_sender(generated_samples, args):
    # File communication to obtain retro-synthesis rate
    with open(args.receiver_file, 'w') as fw:
        fw.write('')
    while(True):
        with open(args.sender_file, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(args.sender_file, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(Chem.MolToSmiles(sample)))
                break
            fcntl.flock(fr, fcntl.LOCK_UN)
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    while(True):
        with open(args.receiver_file, 'r') as fr:
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

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
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

def select_number_with_probabilities():
    """
    Selects a number where:
    - 1 is selected with a probability of 10%
    - 2 is selected with a probability of 80%
    - 3 is selected with a probability of 10%
    
    Returns:
    int: The selected number (1, 2, or 3)
    """
    # Define the choices and their corresponding probabilities
    choices = [1, 2, 3]
    probabilities = [0.2, 0.6, 0.2]
    
    # Use numpy's choice function to select a number based on the defined probabilities
    return np.random.choice(choices, p=probabilities)

def replace_hydrogen_with_asterisk_rdkit(smiles, n):
    """
    Uses RDKit to randomly select n hydrogen atoms in the given SMILES string and replace them with '*'.

    Parameters:
    smiles (str): The SMILES string representing the molecular structure.
    n (int): The number of hydrogen atoms to be replaced with '*'.

    Returns:
    str: The modified SMILES string with n hydrogens replaced by '*'.
    """
    # Convert the SMILES string to an RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Add explicit hydrogens to the molecule
    mol = Chem.AddHs(mol)
    
    # Find all hydrogen atom indices
    h_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    
    # Randomly select n hydrogen atoms, ensuring not to exceed the total number of H atoms
    selected_h_indices = random.sample(h_indices, min(n, len(h_indices)))
    
    # Replace the selected hydrogen atoms with a dummy atom '*'
    for idx in selected_h_indices:
        mol.GetAtomWithIdx(idx).SetAtomicNum(0)  # Atomic number 0 corresponds to the dummy atom '*'
    
    # Convert the modified molecule back to a SMILES string
    modified_smiles = Chem.MolToSmiles(mol)
    
    return modified_smiles

def random_one_two_three():
    """
    Returns 1, 2, or 3 randomly.
    
    Returns:
    int: A random integer among 1, 2, or 3.
    """
    return random.choice([1, 2, 3])

def convert_smi_into_prod_rule(smi):
    print(smi)
    # select starting, ending or normal grammar
    grammar_type = select_number_with_probabilities()
    
    # starting rule
    
    if grammar_type == 1:
        lhs = Hypergraph()
        num_dummy = random_one_two_three()
        rhs = replace_hydrogen_with_asterisk_rdkit(smi,1)
        rhs_mol = Chem.MolFromSmiles(rhs)
        rhs_hg = mol_to_hg_add_dummy(rhs_mol,kekulize=False,add_Hs=False)
        # convert edges with dummy to NT nodes
        

        return ProductionRule(lhs,rhs_hg)
    # middle rule
    elif grammar_type == 2:
        lhs = "**"
        lhs_mol = Chem.MolFromSmiles(lhs)
        lhs_hg = mol_to_hg_add_dummy(lhs_mol,kekulize=False,add_Hs=False)
        num_dummy = random_one_two_three()
        rhs = replace_hydrogen_with_asterisk_rdkit(smi,2)
        rhs_mol = Chem.MolFromSmiles(rhs)
        rhs_hg = mol_to_hg_add_dummy(rhs_mol,kekulize=False,add_Hs=False)
        # convert edges with dummy to NT nodes
        
        rule = ProductionRule(lhs_hg,rhs_hg)
        rule.lhs.remove_edges(["e1"])
        attr_dict = rule.lhs.edge_attr("e0")
        attr_dict["terminal"] = False
        rule.lhs.set_edge_attr("e0", attr_dict)
        
        return rule
    
    # ending rule
    else: 
        print("create ending rule")
        lhs = "**"
        

        lhs_mol = Chem.MolFromSmiles(lhs)
        lhs_hg = mol_to_hg_add_dummy(lhs_mol,kekulize=False,add_Hs=False)
        num_dummy = random_one_two_three()
        rhs = replace_hydrogen_with_asterisk_rdkit(smi,1)
        # rhs = smi
        rhs_mol = Chem.MolFromSmiles(rhs)
        rhs_hg = mol_to_hg_add_dummy(rhs_mol,kekulize=False,add_Hs=False)
        # add adummy to rhs and add NT node
        
        rule = ProductionRule(lhs_hg,rhs_hg)
        rule.lhs.remove_edges(["e1"])
        attr_dict = rule.lhs.edge_attr("e0")
        attr_dict["terminal"] = False
        rule.lhs.set_edge_attr("e0", attr_dict)
        
        
        print(len(rule.rhs.get_all_NT_edges()))
        
        
        return rule
    
    


def convert_frag_into_grammar(smiles_list):
    grammar_init = ProductionRuleCorpus()
    valid_smi_list = []
    for smi in smiles_list:
        try:
            prod_rule = convert_smi_into_prod_rule(smi)
            grammar_init.append(prod_rule)
            valid_smi_list.append(smi)
        except:
            print("error in converting fragments into grammar rule")
            pass
    return grammar_init,valid_smi_list


def learn(smiles_list, args):
    # Create logger
    save_log_path = 'log/log-num_generated_samples{}-{}_grammar_training_{}_grammar_lr_{}_grammar_explore_rate{}_{}'.format(args.num_generated_samples, time.strftime("%Y%m%d-%H%M%S"),args.grammar_training,args.grammar_lr,args.grammar_explore_rate,args.task_name)
    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir('./') if f.endswith('.py')])
    logger = create_logger('global_logger', save_log_path + '/log.txt')
    logger.info('args:{}'.format(pprint.pformat(args)))
    logger = logging.getLogger('global_logger')

    # Initialize dataset & potential function (agent) & optimizer
    if args.select_fragments:
        subgraph_set_init, input_graphs_dict_init = None,None
    else:
        subgraph_set_init, input_graphs_dict_init = data_processing(smiles_list, args.GNN_model_path, args.motif)
    
    agent = Agent(feat_dim=300, hidden_size=args.hidden_size)
    if args.resume:
        assert  os.path.exists(args.resume_path), "Please provide valid path for resuming."
        ckpt = torch.load(args.resume_path)
        agent.load_state_dict(ckpt)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # select fragments
    
    if args.select_fragments:
        frag_grammar,valid_smi_list = convert_frag_into_grammar(smiles_list)
        markov_grammar = markov_grammar_model(frag_grammar,args.grammar_lr,args.grammar_explore_rate,args.grammar_reconstruction_addition,True,valid_smi_list)
        
        
    # Start training
    logger.info('starting\n')
    curr_max_R = 0
    for train_epoch in range(args.max_epoches):
        returns = []
        log_returns = []
        logger.info("<<<<< Epoch {}/{} >>>>>>".format(train_epoch, args.max_epoches))

        if args.batch_training:
            selected_smiles_list = random.sample(smiles_list, args.batch_size)
            subgraph_set_init, input_graphs_dict_init = data_processing(selected_smiles_list, args.GNN_model_path, args.motif)
    
            if args.sa_thres_train_data > 0:
                low_sa_idxs = []
                for i in range(len(selected_smiles_list)):
                    mol = Chem.MolFromSmiles(selected_smiles_list[i])
                    sa_score = sascorer.calculateScore(mol)
                    
                    if sa_score < args.sa_thres_train_data:
                        low_sa_idxs.append(i)
                    
        # MCMC sampling
        
        for num in range(args.MCMC_size):
            grammar_init = ProductionRuleCorpus()
            l_input_graphs_dict = deepcopy(input_graphs_dict_init)
            l_subgraph_set = deepcopy(subgraph_set_init)
            l_grammar = deepcopy(grammar_init)
            
            # Q_table length adjustment
            if args.remove_q_table_thres > 100 and train_epoch > 0:
                if args.remove_q_table_thres < len(markov_grammar.grammar_index):
                    markov_grammar.reduce_q_table_columns(args.remove_q_table_thres)
            
            if args.fragment_ranking_pred:
                total_step = train_epoch * args.MCMC_size + num
                if train_epoch == 0 and num == 0:
                    markov_grammar = markov_grammar_model(l_grammar,args.grammar_lr,args.grammar_explore_rate,args.grammar_reconstruction_addition)
                    iter_num, l_grammar, l_input_graphs_dict = MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num, args, markov_model=markov_grammar,gammar=args.fragment_ranking_pred_gammar**total_step,total_steps=total_step)
                else:
                    iter_num, l_grammar, l_input_graphs_dict = MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num, args, markov_model=markov_grammar,gammar=args.fragment_ranking_pred_gammar**total_step,total_steps=total_step)
            elif args.select_fragments:
                total_step = train_epoch * args.MCMC_size + num
            
            else:
                iter_num, l_grammar, l_input_graphs_dict = MCMC_sampling(agent, l_input_graphs_dict, l_subgraph_set, l_grammar, num, args)
            
            
            
            
            # Grammar evaluation
            

            ## Grammar Training
            if args.grammar_training:
                # model initialization
                
                if train_epoch == 0 and num == 0:
                    markov_grammar = markov_grammar_model(l_grammar,args.grammar_lr,args.grammar_explore_rate,args.grammar_reconstruction_addition)
                
                markov_grammar.adjust_q_table_columns(l_grammar)
                logger.info(f"q_table shape: {markov_grammar.q_table.shape}")

                if args.without_reconstruction:
                    pass
                else:
                    if args.sa_thres_train_data > 0:
                        markov_grammar.train_markov_grammar_by_reconstruction(l_input_graphs_dict,low_sa_idxs)
                    else:
                        markov_grammar.train_markov_grammar_by_reconstruction(l_input_graphs_dict)
                markov_grammar.update_tmp_current_grammar_idx(l_grammar)
                for i in range(args.grammar_training_round):
                    diversity,syn = markov_grammar.train_markov_grammar(l_grammar,args,args.grammar_training_round_sample)
                    logger.info(f"markov model training results=== diversity: {diversity}, synthesizability: {syn}")
                if args.fragment_ranking_pred:
                    R_ind = diversity + 2 * syn
                    logger.info("======Sample {} returns {}=======:".format(total_step, R_ind))
                    # Save ckpt
                    if R_ind > curr_max_R or total_step == (args.max_epoches * args.MCMC_size - 1) or total_step%50 == 0:
                        torch.save(agent.state_dict(), os.path.join(save_log_path, 'epoch_agent_{}_{}.pkl'.format(total_step, R_ind)))
                        with open('{}/epoch_grammar_{}_{}.pkl'.format(save_log_path, total_step, R_ind), 'wb') as outp:
                            pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                        with open('{}/epoch_input_graphs_{}_{}.pkl'.format(save_log_path, total_step, R_ind), 'wb') as outp:
                            pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                        curr_max_R = R_ind
                        
                        markov_grammar_filename = '{}/epoch_markov_grammar_{}_{}.pkl'.format(save_log_path, total_step, R_ind)
                        markov_grammar.save(markov_grammar_filename)
                else:
                    eval_metric = evaluate_by_trained_grammar(l_grammar, args, markov_grammar ,metrics=['diversity', 'syn'])
                    logger.info("eval_metrics: {}".format(eval_metric))
                    # Record metrics
                    R = eval_metric['diversity'] + 2 * eval_metric['syn']
                    R_ind = copy.deepcopy(R)

            elif args.select_fragments:
                if train_epoch == 0 and num == 0:
                    markov_grammar.update_tmp_current_grammar_idx(frag_grammar)
                for i in range(args.grammar_training_round):
                    diversity,syn = markov_grammar.train_markov_grammar(frag_grammar,args,args.grammar_training_round_sample)
                    logger.info(f"markov model training results=== diversity: {diversity}, synthesizability: {syn}")
                
                R_ind = diversity + 2 * syn
                logger.info("======Sample {} returns {}=======:".format(total_step, R_ind))
                # Save ckpt
                if R_ind > curr_max_R or total_step == (args.max_epoches * args.MCMC_size - 1) or total_step%50 == 0:
                    torch.save(agent.state_dict(), os.path.join(save_log_path, 'epoch_agent_{}_{}.pkl'.format(total_step, R_ind)))
                    with open('{}/epoch_grammar_{}_{}.pkl'.format(save_log_path, total_step, R_ind), 'wb') as outp:
                        pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                    with open('{}/epoch_input_graphs_{}_{}.pkl'.format(save_log_path, total_step, R_ind), 'wb') as outp:
                        pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                    curr_max_R = R_ind
                    
                    markov_grammar_filename = '{}/epoch_markov_grammar_{}_{}.pkl'.format(save_log_path, total_step, R_ind)
                    markov_grammar.save(markov_grammar_filename)
            
                
            else:
                eval_metric = evaluate(l_grammar, args, metrics=['diversity', 'syn'])
                logger.info("eval_metrics: {}".format(eval_metric))
                # Record metrics
                R = eval_metric['diversity'] + 2 * eval_metric['syn']
                R_ind = copy.deepcopy(R)
            
            if args.fragment_ranking_pred or args.select_fragments:
                pass
            else:
                returns.append(R)
                log_returns.append(eval_metric)
                logger.info("======Sample {} returns {}=======:".format(num, R_ind))
                # Save ckpt
                if R_ind > curr_max_R:
                    torch.save(agent.state_dict(), os.path.join(save_log_path, 'epoch_agent_{}_{}.pkl'.format(train_epoch, R_ind)))
                    with open('{}/epoch_grammar_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                        pickle.dump(l_grammar, outp, pickle.HIGHEST_PROTOCOL)
                    with open('{}/epoch_input_graphs_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind), 'wb') as outp:
                        pickle.dump(l_input_graphs_dict, outp, pickle.HIGHEST_PROTOCOL)
                    curr_max_R = R_ind
                    
                    if args.grammar_training:
                        markov_grammar_filename = '{}/epoch_markov_grammar_{}_{}.pkl'.format(save_log_path, train_epoch, R_ind)
                        markov_grammar.save(markov_grammar_filename)
        if args.fragment_ranking_pred or args.select_fragments:
            pass
        else:
            # Calculate loss
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) # / (returns.std() + eps)
            assert len(returns) == len(list(agent.saved_log_probs.keys()))
            policy_loss = torch.tensor([0.])
            for sample_number in agent.saved_log_probs.keys():
                max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
                for iter_num_key in agent.saved_log_probs[sample_number].keys():
                    log_probs = agent.saved_log_probs[sample_number][iter_num_key]
                    for log_prob in log_probs:
                        policy_loss += (-log_prob * args.gammar ** (max_iter_num - iter_num_key) * returns[sample_number]).sum()

            # Back Propogation and update
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            agent.saved_log_probs.clear()

            # Log
            logger.info("Loss: {}".format(policy_loss.clone().item()))
            eval_metrics = {}
            for r in log_returns:
                for _key in r.keys():
                    if _key not in eval_metrics:
                        eval_metrics[_key] = []
                    eval_metrics[_key].append(r[_key])
            mean_evaluation_metrics = ["{}: {}".format(_key, np.mean(eval_metrics[_key])) for _key in eval_metrics]
            logger.info("Mean evaluation metrics: {}".format(', '.join(mean_evaluation_metrics)))


        # log Q-table 
        if args.grammar_training:
            logger.info(f"Q_table: {markov_grammar.get_q_table()}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCMC training')
    parser.add_argument('--training_data', type=str, default="./datasets/isocyanates.txt", help="file name of the training data")
    parser.add_argument('--GNN_model_path', type=str, default="./GCN/model_gin/supervised_contextpred.pth", help="file name of the pretrained GNN model")
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden size of the potential function")
    parser.add_argument('--max_epoches', type=int, default=50, help="maximal training epoches")
    parser.add_argument('--num_generated_samples', type=int, default=100, help="number of generated samples to evaluate grammar")
    parser.add_argument('--MCMC_size', type=int, default=5, help="sample number of each step of MCMC")
    parser.add_argument('--learning_rate', type=int, default=1e-2, help="learning rate")
    parser.add_argument('--gammar', type=float, default=0.99, help="discount factor")
    parser.add_argument('--motif', action="store_true", default=False, help="use motif as the basic building block for polymer dataset")
    parser.add_argument('--sender_file', type=str, default="generated_samples.txt", help="file name of the generated samples")
    parser.add_argument('--receiver_file', type=str, default="output_syn.txt", help="file name of the output file of Retro*")
    parser.add_argument('--resume', action="store_true", default=False, help="resume model")
    parser.add_argument('--resume_path', type=str, default='', help="resume path")
    
    # grammar training
    parser.add_argument('--grammar_training', action="store_true", default=False, help="train grammar using Marcov model")
    parser.add_argument('--grammar_training_round', type=int, default=5, help="grammar training epochs")
    parser.add_argument('--grammar_training_round_sample', type=int, default=100, help="grammar training samples")
    parser.add_argument('--grammar_lr', type=float, default=0.01, help="learning rate for grammar training")
    parser.add_argument('--grammar_explore_rate', type=float, default=0.2, help="exploration rate for grammar training")
    parser.add_argument('--grammar_reconstruction_addition', type=float, default=0.5, help="reconstruction addition reward for grammar training")
    
    parser.add_argument('--without_reconstruction', action="store_true", default=False, help="train grammar without using reconstruction loss")
    # without retrostar
    parser.add_argument('--without_retro', action="store_true", default=False, help="without retro star process")
    parser.add_argument('--sa_score', action="store_true", default=False, help="without retro star process")
    
    parser.add_argument('--sa_thres_train_data', type=float, default=0, help="without retro star process")
    
    
    # predict bond cutting by fragment ranking
    parser.add_argument('--fragment_ranking_pred', action="store_true", default=False, help="predict bond cutting based on q table")
    parser.add_argument('--fragment_ranking_pred_gammar', type=float, default=0.99, help="discount factor")
    parser.add_argument('--random_warm_up', type=int, default=5, help="random cut")
    
    parser.add_argument('--task_name', type=str, help="task name")
    
    # batch training 
    parser.add_argument('--batch_training', action="store_true", default=False, help="without retro star process")
    parser.add_argument('--batch_size', type=int, default=5, help="random cut")
    
    
    parser.add_argument('--remove_q_table_thres', type=int, default=100, help="random cut")
    
    
    # select fragments
    parser.add_argument('--select_fragments', action="store_true", default=False, help="without retro star process")
    parser.add_argument('--select_fragments_num_frag', type=int, default=2000, help="without retro star process")
    
    
    args = parser.parse_args()

    # Get raw training data
    assert os.path.exists(args.training_data), "Please provide valid path of training data."
    # Remove duplicated molecules
    with open(args.training_data, 'r') as fr:
        lines = fr.readlines()
        mol_sml = []
        for line in lines:
            if not (line.strip() in mol_sml):
                mol_sml.append(line.strip())

    # Clear the communication files for Retro*
    with open(args.sender_file, 'w') as fw:
        fw.write('')
    with open(args.receiver_file, 'w') as fw:
        fw.write('')
    
    # Grammar learning
    learn(mol_sml[:args.select_fragments_num_frag], args)

