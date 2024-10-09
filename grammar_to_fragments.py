import argparse
from cmath import inf
import pickle5 as pickle
from os import listdir
from rdkit import Chem
import os
import time
import pprint
import logging
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Fragments
import copy
import random

from private import *
from grammar_generation import random_produce
from retro_star_listener import *

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def select_grammar_id_based_on_chosen_rules(top_k_indices,markov_grammar):
    grammar_list_q_table = copy.deepcopy(markov_grammar.grammar_index)
    
    
    # print("current_grammar_idx_q_table:",current_grammar_idx_q_table)
    q_score_list = []
    
    for g_idx in range(len(grammar_list_q_table)):
        if g_idx in top_k_indices:
            q_score_list.append(-1000)
        else:
            q_score = 0
            for current_g_idx in top_k_indices:
                if current_g_idx == None:
                    pass
                else:
                    q_score += markov_grammar.q_table[current_g_idx][g_idx]
                    q_score += markov_grammar.q_table[g_idx][current_g_idx]
                
            q_score_list.append(q_score)
            
    sorted_indices = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)[:args.add_top_k]
    
    return sorted_indices[0]
    


def get_additional_top_k_grammar(markov_grammar,grammar,args):
    grammar_list_q_table = copy.deepcopy(markov_grammar.grammar_index)
    
    if args.topk_from_scratch:
        grammar.prod_rule_list = []
        markov_grammar.tmp_current_grammar_idx = []
    
    current_grammar_idx_q_table = markov_grammar.tmp_current_grammar_idx
    print("current_grammar_idx_q_table:",current_grammar_idx_q_table)
    q_score_list = []
    
    for g_idx in range(len(grammar_list_q_table)):
        if g_idx in current_grammar_idx_q_table:
            q_score_list.append(-1000)
        else:
            q_score = 0
            for current_g_idx in current_grammar_idx_q_table:
                if current_g_idx == None:
                    pass
                else:
                    q_score += markov_grammar.q_table[current_g_idx][g_idx]
                    q_score += markov_grammar.q_table[g_idx][current_g_idx]
                
            q_score_list.append(q_score)
            
    top_k_indices = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)[:args.add_top_k]
    if args.add_top_k_random:
        top_k_indices = random.sample(range(len(q_score_list)), args.add_top_k)
    
    if args.add_top_k_starting_ending > 0:
        top_k_indices = []
        sorted_grammar_idx = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)
        
        top_k_starting = []
        top_k_ending = []
        
        for grammar_idx in sorted_grammar_idx:
            grammar_rule = markov_grammar.grammar_index[grammar_idx]
            if grammar_rule.is_start_rule == True and len(top_k_starting) < args.add_top_k_starting_ending:
                top_k_starting.append(grammar_idx)
            if grammar_rule.is_ending == True and len(top_k_ending) < args.add_top_k_starting_ending:
                top_k_ending.append(grammar_idx)
            if grammar_rule.is_start_rule == False and grammar_rule.is_ending == False and len(top_k_indices) < ( args.add_top_k - 2 * args.add_top_k_starting_ending ):
                top_k_indices.append(grammar_idx)
        
        top_k_indices +=   top_k_starting  
        top_k_indices +=   top_k_ending  
        
        if args.top_k_based_on_chosen_ones:
            top_k_indices = []
            top_k_indices +=   top_k_starting  
            top_k_indices +=   top_k_ending  
            
            while len(top_k_indices) < args.add_top_k:
                top_k_indices.append(select_grammar_id_based_on_chosen_rules(top_k_indices,markov_grammar))
            
    
    if args.topk_from_scratch == False and args.top_k_based_on_chosen_ones == True:
        top_k_indices = copy.deepcopy(markov_grammar.tmp_current_grammar_idx)
        original_grammar_rule_len = len(top_k_indices)
        while (len(top_k_indices) - original_grammar_rule_len ) < args.add_top_k:
            top_k_indices.append(select_grammar_id_based_on_chosen_rules(top_k_indices,markov_grammar))
        top_k_indices_copy = copy.deepcopy(top_k_indices)
        top_k_indices = [elem for elem in top_k_indices_copy if elem not in markov_grammar.tmp_current_grammar_idx   ]  
            
    
    print("top_k_indices: ", top_k_indices)
    markov_grammar.tmp_current_grammar_idx += top_k_indices
    print("current_grammar_idx_q_table:",markov_grammar.tmp_current_grammar_idx)
    return [grammar_list_q_table[i] for i in top_k_indices]

import re
from rdkit import Chem

def hide_hydrogens_in_smiles(mol):
    """Return the SMILES string of the molecule with hidden hydrogens."""
    # Create a copy of the molecule with hydrogens hidden
    mol_no_h = Chem.RemoveHs(mol)
    # Convert the molecule to SMILES with kekulization and explicit hydrogen options
    smiles = Chem.MolToSmiles(mol_no_h, kekuleSmiles=True, allHsExplicit=False)
    return smiles

def get_neighboring_atoms(mol, atom_idx):
    """Return a list of atom indices neighboring the given atom index."""
    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        raise ValueError("Invalid atom index")
    
    neighboring_atoms = []
    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        neighboring_atoms.append(neighbor)
    return neighboring_atoms

def convert_smiles_with_dummy(smiles):
    # Parse the input SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol_del = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles('*'))
    if mol is None:
        return None, None

    # Replace the dummy atom with a regular atom (e.g., carbon)
    
    Chem.Kekulize(mol)
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1) 
            neighboring_atoms = get_neighboring_atoms(mol,atom.GetIdx())
            for a in neighboring_atoms:
                a.SetAtomMapNum( a.GetIdx() +1 )

    # Generate SMILES without dummy atoms
    smiles_no_dummy = Chem.RemoveHs(mol)
    
    smiles_no_dummy = Chem.MolToSmiles(smiles_no_dummy)
    
    def convert_integer_to_1(smiles):
        # Define a regular expression pattern to match integers following a colon
        pattern = r'(?<=:)\d+'
        # Replace any matched integers with '1'
        converted_smiles = re.sub(pattern, '1', smiles)
        return converted_smiles
    
    

    return Chem.MolToSmiles(mol_del),convert_integer_to_1(smiles_no_dummy)


def load_and_generate_mols(args):
    
    
    ckpt_list = listdir(args.model_folder)
    max_R = 0
    max_R_ckpt = None
    for ckpt in ckpt_list:
        if 'epoch_grammar' in ckpt:
            print("ckpt = ",ckpt)
            curr_R = float(ckpt.split('_')[3][:-4])
            epochs_num = int(ckpt.split('_')[2])
            if epochs_num <= args.early_stopping:
                if curr_R > max_R:
                    max_R = curr_R
                    max_R_ckpt = ckpt
            if args.final_model:
                max_R = curr_R
                max_R_ckpt = ckpt
        if 'best_grammar' in ckpt:
            print("ckpt = ",ckpt)
            curr_R = float(ckpt.split('_')[4][:-4])
            epochs_num = int(ckpt.split('_')[3])
            if epochs_num <= args.early_stopping:
                if curr_R > max_R:
                    max_R = curr_R
                    max_R_ckpt = ckpt
    print('loading {}'.format(max_R_ckpt))
    with open('{}/{}'.format(args.model_folder, max_R_ckpt), 'rb') as fr:
        grammar = pickle.load(fr)
    if args.grammar_training:
        max_R = 0
        max_R_ckpt = None
        for ckpt in ckpt_list:
            if 'epoch_markov' in ckpt:
                print("ckpt = ",ckpt)
                curr_R = float(ckpt.split('_')[4][:-4])
                epochs_num = int(ckpt.split('_')[3])
                if epochs_num <= args.early_stopping:
                    if curr_R > max_R:
                        max_R = curr_R
                        max_R_ckpt = ckpt
                if args.final_model:
                    max_R = curr_R
                    max_R_ckpt = ckpt
        print('loading {}'.format(max_R_ckpt))
        with open('{}/{}'.format(args.model_folder, max_R_ckpt), 'rb') as fr:
            markov_grammar = pickle.load(fr)
            print("tmp_current_grammar_idx: ",markov_grammar.tmp_current_grammar_idx)
            
    if args.grammar_training and args.add_top_k > 0:
        additional_top_k_grammar =  get_additional_top_k_grammar(markov_grammar,grammar,args)
        for grammar_i in additional_top_k_grammar:
            grammar.prod_rule_list.append(grammar_i)
    
    
    # convert grammar to fragments and save them
    
    smi_list = []
    smi_with_num_list = []
    smi_with_dummy_list = []
    
    for grammar_i in grammar.prod_rule_list:
        try:
            hg = grammar_i.rhs
            mol = hg_to_mol(hg)
            smi_with_dummy = Chem.MolToSmiles(mol)
            smi_with_dummy_explicit = Chem.MolToSmiles(mol,allBondsExplicit=True)
            
            smi,smi_num = convert_smiles_with_dummy(smi_with_dummy)
            smi_list.append(smi)
            smi_with_dummy_list.append(smi_with_dummy_explicit)
            smi_with_num_list.append(smi_num)
            
        except:
            print("error in making smiles")
            
    def save_smiles_pairs(smiles_list1, smiles_list2, filename):
        with open(filename, 'w') as file:
            # Iterate over both lists simultaneously using zip
            for smiles1, smiles2 in zip(smiles_list1, smiles_list2):
                # Write the pair of SMILES to the file separated by a space
                file.write(f"{smiles1} {smiles2}\n")
                
    def sort_and_save_smiles(smiles_list, filename):
        # Sort the smiles based on the number of atoms in ascending order
        unique_smiles = list(set(smiles_list))
        sorted_smiles = sorted(unique_smiles, key=lambda x: get_num_atoms(x))

        # Save the sorted smiles to the specified file
        with open(filename, 'w') as file:
            for smile in sorted_smiles:
                if len(smile) >= 1:
                    file.write(smile + '\n')
    
    def no_sort_and_save_smiles(smiles_list, filename):
        # Sort the smiles based on the number of atoms in ascending order
        unique_smiles = list(set(smiles_list))
        
        # Save the sorted smiles to the specified file
        with open(filename, 'w') as file:
            for smile in unique_smiles:
                if len(smile) >= 1:
                    file.write(smile + '\n')

    def get_num_atoms(smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                return mol.GetNumAtoms()
            else:
                return float('inf')  # Return infinity if the SMILES cannot be converted to a molecule
        except:
            return float('inf')  # Return infinity if there's any error during conversion

                
    file_name = args.model_folder + "/fragments.txt"
        
    save_smiles_pairs(smi_list,smi_with_num_list,file_name)
    
    file_name = args.model_folder + "/fragments_micam.txt"
        
    save_smiles_pairs(smi_list,smi_with_dummy_list,file_name)
   
    file_name = args.model_folder + "/fragments_micam_merge_operation.txt"
    
    sort_and_save_smiles(smi_list,file_name)
    
    file_name = args.model_folder + "/fragments_micam_merge_operation_no_sort.txt"
    
    no_sort_and_save_smiles(smi_list,file_name)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCMC training')
    parser.add_argument('--model_folder', type=str, help="folder name of the trained model")
    
    # grammar training
    parser.add_argument('--grammar_training', action="store_true", default=False, help="train grammar using Marcov model")
    parser.add_argument('--early_stopping',  type=int, default=100, help="train grammar using Marcov model")
    
    parser.add_argument('--add_top_k',  type=int, default=0, help="add another top k from q table")
    parser.add_argument('--add_top_k_random', action="store_true", default=False, help="add another top k from q table randomly for baseline comparison")
    
    parser.add_argument('--topk_from_scratch', action="store_true", default=False, help="add another top k from q table randomly for baseline comparison")
    
    
    parser.add_argument('--add_top_k_starting_ending',  type=int, default=0, help="add another top k from q table")
    
    parser.add_argument('--final_model', action="store_true", default=False, help="add another top k from q table randomly for baseline comparison")
     
    parser.add_argument('--task_name', type=str, help="task name")
    
     
    
    args = parser.parse_args()
    # load the best model and generate 10k samples
    load_and_generate_mols(args)
     