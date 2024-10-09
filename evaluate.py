import argparse
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

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def select_grammar_id_based_on_chosen_rules(top_k_indices, markov_grammar):
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

    sorted_indices = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)[: args.add_top_k]

    return sorted_indices[0]


def select_elements_by_indexes(input_list, indexes):
    # Initialize an empty list to store the selected elements
    selected_elements = []

    # Iterate over the indexes list
    for index in indexes:
        # Check if the index is within the bounds of the input_list
        if 0 <= index < len(input_list):
            # Append the element at the current index to the selected_elements list
            selected_elements.append(input_list[index])
        else:
            # If the index is out of bounds, append None or handle appropriately
            selected_elements.append(None)

    return selected_elements


def get_additional_top_k_grammar(markov_grammar, grammar, args):
    grammar_list_q_table = copy.deepcopy(markov_grammar.grammar_index)

    if args.topk_from_scratch:
        grammar.prod_rule_list = []
        markov_grammar.tmp_current_grammar_idx = []

    current_grammar_idx_q_table = markov_grammar.tmp_current_grammar_idx
    print("current_grammar_idx_q_table:", current_grammar_idx_q_table)
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

    top_k_indices = sorted(range(len(q_score_list)), key=lambda i: q_score_list[i], reverse=True)[: args.add_top_k]
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
            if (
                grammar_rule.is_start_rule == False
                and grammar_rule.is_ending == False
                and len(top_k_indices) < (args.add_top_k - 2 * args.add_top_k_starting_ending)
            ):
                top_k_indices.append(grammar_idx)

        top_k_indices += top_k_starting
        top_k_indices += top_k_ending

        if args.top_k_based_on_chosen_ones:
            top_k_indices = []
            top_k_indices += top_k_starting
            top_k_indices += top_k_ending

            while len(top_k_indices) < args.add_top_k:
                top_k_indices.append(select_grammar_id_based_on_chosen_rules(top_k_indices, markov_grammar))

    if args.topk_from_scratch == False and args.top_k_based_on_chosen_ones == True:
        top_k_indices = copy.deepcopy(markov_grammar.tmp_current_grammar_idx)
        original_grammar_rule_len = len(top_k_indices)
        while (len(top_k_indices) - original_grammar_rule_len) < args.add_top_k:
            top_k_indices.append(select_grammar_id_based_on_chosen_rules(top_k_indices, markov_grammar))
        top_k_indices_copy = copy.deepcopy(top_k_indices)
        top_k_indices = [elem for elem in top_k_indices_copy if elem not in markov_grammar.tmp_current_grammar_idx]

    if args.select_frag:
        select_frag_list = select_elements_by_indexes(markov_grammar.frag_list, top_k_indices)
        return select_frag_list

    print("top_k_indices: ", top_k_indices)
    markov_grammar.tmp_current_grammar_idx += top_k_indices
    print("current_grammar_idx_q_table:", markov_grammar.tmp_current_grammar_idx)
    return [grammar_list_q_table[i] for i in top_k_indices]


def load_and_generate_mols(args):
    generated_mols = []
    expr_name = args.expr_name
    print("dealing with {}".format(expr_name))
    ckpt_list = listdir(args.model_folder)
    max_R = 0
    max_R_ckpt = None
    for ckpt in ckpt_list:
        if "epoch_grammar" in ckpt:
            print("ckpt = ", ckpt)
            curr_R = float(ckpt.split("_")[3][:-4])
            epochs_num = int(ckpt.split("_")[2])
            if epochs_num <= args.early_stopping:
                if curr_R > max_R:
                    max_R = curr_R
                    max_R_ckpt = ckpt
            if args.final_model:
                max_R = curr_R
                max_R_ckpt = ckpt
        if "best_grammar" in ckpt:
            print("ckpt = ", ckpt)
            curr_R = float(ckpt.split("_")[4][:-4])
            epochs_num = int(ckpt.split("_")[3])
            if epochs_num <= args.early_stopping:
                if curr_R > max_R:
                    max_R = curr_R
                    max_R_ckpt = ckpt
    print("loading {}".format(max_R_ckpt))
    with open("{}/{}".format(args.model_folder, max_R_ckpt), "rb") as fr:
        grammar = pickle.load(fr)
    if args.grammar_training:
        max_R = 0
        max_R_ckpt = None
        for ckpt in ckpt_list:
            if "epoch_markov" in ckpt:
                print("ckpt = ", ckpt)
                curr_R = float(ckpt.split("_")[4][:-4])
                epochs_num = int(ckpt.split("_")[3])
                if epochs_num <= args.early_stopping:
                    if curr_R > max_R:
                        max_R = curr_R
                        max_R_ckpt = ckpt
                if args.final_model:
                    max_R = curr_R
                    max_R_ckpt = ckpt
        print("loading {}".format(max_R_ckpt))
        with open("{}/{}".format(args.model_folder, max_R_ckpt), "rb") as fr:
            markov_grammar = pickle.load(fr)
            print("tmp_current_grammar_idx: ", markov_grammar.tmp_current_grammar_idx)

    if args.select_frag:
        select_frag = get_additional_top_k_grammar(markov_grammar, grammar, args)

        def save_smiles_to_file(smiles_list, file_name):
            # Open the file with write mode
            with open(file_name, "w") as file:
                # Iterate through each SMILES string in the list
                for smiles in smiles_list:
                    # Write each SMILES string to the file followed by a newline
                    file.write(smiles + "\n")

        save_smiles_to_file(select_frag, args.select_frag_filename)
        return None, None
    if args.grammar_training and args.add_top_k > 0:
        additional_top_k_grammar = get_additional_top_k_grammar(markov_grammar, grammar, args)
        for grammar_i in additional_top_k_grammar:
            grammar.prod_rule_list.append(grammar_i)

    if args.save_grammar_rule_png:
        save_log_path = "log/log-evaluate_{}-{}_grammar_training_{}_expr_name_{}_{}".format(
            args.num_generated_samples, time.strftime("%Y%m%d-%H%M%S"), args.grammar_training, args.expr_name, args.task_name
        )
        create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir("./") if f.endswith(".py")])

        counter = 0
        for grammar_rule_i in grammar.prod_rule_list:
            grammar_rule_i.rhs.draw(file_path=save_log_path + "/rhs" + str(counter), with_node=True, with_edge_name=True)
            grammar_rule_i.lhs.draw(file_path=save_log_path + "/lhs" + str(counter), with_node=True, with_edge_name=True)

            counter += 1

        return None, None

    for i in range(args.num_generated_samples):
        if args.grammar_training and args.random_combination == False:
            mol, iter_num, _ = markov_grammar.produce(
                grammar, args.uniform_smoothing_e, args.a_ending_factor, args.top_k_selection, args.random_iter_start
            )
            generated_mols.append(mol)
        else:
            mol, _ = random_produce(grammar)
            generated_mols.append(mol)

    return generated_mols, grammar


def calculate_validity_percentage(mol_list):
    valid_count = sum(1 for mol in mol_list if mol is not None and Chem.MolToSmiles(mol) != "")
    total_count = len(mol_list)
    if total_count == 0:
        return 0.0  # Avoid division by zero
    percentage = (valid_count / total_count) * 100.0
    return percentage


def calculate_uniqueness_percentage(mol_list):
    unique_smiles = set()
    for mol in mol_list:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            unique_smiles.add(smiles)
    total_count = len(mol_list)
    unique_count = len(unique_smiles)
    if total_count == 0:
        return 0.0  # Avoid division by zero
    percentage = (unique_count / total_count) * 100.0
    return percentage


def calculate_rs(generated_mols):
    syn = Synthesisability()
    syn_status = []
    for sample_smi in generated_mols:
        print("predict synthesizability")
        try:

            result = syn.planner.plan(Chem.MolToSmiles(sample_smi))
        except:
            result = None
        result = False if result is None else True
        syn_status.append(result)
    return sum(syn_status) / len(syn_status)


def calculate_tanimoto_distance(mol, smiles):
    """
    Calculate the Tanimoto distance between a molecule and a SMILES string.
    """
    mol1 = mol
    mol2 = Chem.MolFromSmiles(smiles)

    if mol1 is not None and mol2 is not None:
        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        distance = 1 - DataStructs.TanimotoSimilarity(fp1, fp2)
        return distance

    return 0  # Return None if either of the input SMILES is invalid


def calculate_min_chamfer_distance(mol_list, smiles_list):
    """
    Calculate the Chamfer distance between a list of molecules and a list of SMILES strings.
    """
    total_distance = 0
    num_mols = len(mol_list)
    for mol in mol_list:
        min_distance = float("inf")  # Initialize the minimum distance to infinity
        for smiles in smiles_list:
            distance = calculate_tanimoto_distance(mol, smiles)
            if distance < min_distance:
                min_distance = distance
        total_distance += min_distance
    chamfer_distance = total_distance / num_mols
    return chamfer_distance


def calculate_avg_chamfer_distance(mol_list, smiles_list):
    """
    Calculate the Chamfer distance between a list of molecules and a list of SMILES strings.
    """
    total_distance = 0
    num_mols = len(mol_list)
    num_smi = len(smiles_list)
    for mol in mol_list:
        total_distance_mol = 0
        for smiles in smiles_list:
            distance = calculate_tanimoto_distance(mol, smiles)
            total_distance_mol += distance
        total_distance += total_distance_mol / num_smi
    chamfer_distance = total_distance / num_mols
    return chamfer_distance


def calculate_coverage_chamfer_distance(mol_list, smiles_list):
    """
    Calculate the Chamfer distance between a list of molecules and a list of SMILES strings.
    """
    total_distance = 0
    num_mols = len(mol_list)
    for smiles in smiles_list:
        min_distance = float("inf")  # Initialize the minimum distance to infinity
        for mol in mol_list:
            distance = calculate_tanimoto_distance(mol, smiles)
            if distance < min_distance and distance > 0:
                min_distance = distance
        total_distance += min_distance
    chamfer_distance = total_distance / num_mols
    return 1 - chamfer_distance


def is_isocyanate(mol):
    """
    Check if a molecule belongs to the isocyanate monomer class.
    """
    if mol is None:
        return False
    # Define the SMARTS pattern for the isocyanate functional group
    isocyanate_smarts = Chem.MolFromSmarts("N=C=O")
    # Check if the molecule contains the isocyanate functional group
    return mol.HasSubstructMatch(isocyanate_smarts)


def percentage_isocyanates(mol_list):
    """
    Calculate the percentage of molecules in the list belonging to the isocyanate monomer class.
    """
    total_count = len(mol_list)
    if total_count == 0:
        return 0.0
    isocyanate_count = sum(1 for mol in mol_list if is_isocyanate(mol))
    percentage = (isocyanate_count / total_count) * 100.0
    return percentage


def is_acrylate(mol):
    """
    Check if a molecule belongs to the acrylate monomer class.
    """
    if mol is None:
        return False
    # Define the SMARTS pattern for the acrylate functional group
    acrylate_smarts = Chem.MolFromSmarts("C=C[C,C]=O")
    # Check if the molecule contains the acrylate functional group
    return mol.HasSubstructMatch(acrylate_smarts)


def percentage_acrylates(mol_list):
    """
    Calculate the percentage of molecules in the list belonging to the acrylate monomer class.
    """
    total_count = len(mol_list)
    if total_count == 0:
        return 0.0
    acrylate_count = sum(1 for mol in mol_list if is_acrylate(mol))
    percentage = (acrylate_count / total_count) * 100.0
    return percentage


def is_diol_or_diamine(mol):
    """
    Check if a molecule belongs to the diol or diamine monomer class.
    """
    try:
        oh_count = Fragments.fr_Ar_OH(mol) + Fragments.fr_Al_OH(mol)
        nh_count = Fragments.fr_NH2(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH0(mol)
        total_chain_bond_count = oh_count + nh_count
    except:
        return False

    if total_chain_bond_count >= 2:
        return True
    return False


def percentage_diols_diamines(mol_list):
    """
    Calculate the percentage of molecules in the list belonging to the diol or diamine monomer class.
    """
    total_count = len(mol_list)
    if total_count == 0:
        return 0.0
    diol_diamine_count = sum(1 for mol in mol_list if is_diol_or_diamine(mol))
    percentage = (diol_diamine_count / total_count) * 100.0
    return percentage


def calculate_membership(generated_mols, args):
    membership_score = 0

    if args.expr_name == "chain_extenders":
        membership_score = percentage_diols_diamines(generated_mols)
    if args.expr_name == "isocyanates":
        membership_score = percentage_isocyanates(generated_mols)

    if args.expr_name == "acrylates":
        membership_score = percentage_acrylates(generated_mols)

    return membership_score


def percentage_mols_not_in_smiles(mol_list, smiles_list):
    """
    Calculate the percentage of molecules in mol_list that are not included in smiles_list.
    """
    if not mol_list:
        return 0.0

    # Convert the list of SMILES strings to a set for faster membership checking
    smiles_set = set(smiles_list)

    # Count the number of molecules not included in smiles_list
    try:
        count_not_in_smiles = sum(1 for mol in mol_list if Chem.MolToSmiles(mol) not in smiles_set)
    except:
        count_not_in_smiles = 0
        for mol in mol_list:
            try:
                mol_smi = Chem.MolToSmiles(mol)
                if mol_smi not in smiles_set:
                    count_not_in_smiles += 1
            except:
                pass
    # Calculate the percentage
    percentage = (count_not_in_smiles / len(mol_list)) * 100.0
    return percentage


def calculate_useful_mol_percentage(generated_mols, mol_sml, without_membership=False):
    total_count = len(generated_mols)

    unique_smiles = set()
    all_smi = []
    for mol in generated_mols:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            unique_smiles.add(smiles)
            all_smi.append(smiles)

    useful_smi_list = []
    syn = Synthesisability()
    for uni_smi in unique_smiles:
        if uni_smi not in mol_sml:
            # novelty
            try:
                result = syn.planner.plan(uni_smi)
            except:
                result = None
            result = False if result is None else True

            if result == True:
                membership_score = False
                if args.expr_name == "chain_extenders":
                    membership_score = is_diol_or_diamine(Chem.MolFromSmiles(uni_smi))
                if args.expr_name == "isocyanates":
                    membership_score = is_acrylate(Chem.MolFromSmiles(uni_smi))

                if args.expr_name == "acrylates":
                    membership_score = is_acrylate(Chem.MolFromSmiles(uni_smi))

                if membership_score == True:
                    useful_smi_list.append(uni_smi)
                elif without_membership:
                    useful_smi_list.append(uni_smi)

    num_useful_mols = len(useful_smi_list)

    all_useful_smi = [smi for smi in all_smi if smi in useful_smi_list]

    useful_validity = len(all_useful_smi) / total_count
    if len(all_useful_smi) == 0:
        useful_uniqueness = 0
    else:
        useful_uniqueness = len(useful_smi_list) / len(all_useful_smi)

    div = InternalDiversity()
    all_useful_mols = [Chem.MolFromSmiles(smi) for smi in all_useful_smi]
    if len(all_useful_smi) == 0:
        useful_diversity = 0
    else:
        useful_diversity = div.get_diversity(all_useful_mols)

    if len(all_useful_smi) == 0:
        useful_chamfer = 0
    else:
        useful_chamfer = calculate_avg_chamfer_distance(all_useful_mols, mol_sml)

    return num_useful_mols, useful_validity, useful_uniqueness, useful_diversity, useful_chamfer, useful_smi_list


def save_smiles_to_file(mol_list, filename):
    """
    Save a list of RDKit molecules to a text file containing SMILES strings.

    Parameters:
        mol_list (list): A list of RDKit molecules.
        filename (str): The name of the text file to save the SMILES strings to.
    """

    # Open the file in write mode
    with open(filename, "w") as f:
        # Write each SMILES string to the file
        for mol in mol_list:
            try:
                smiles = Chem.MolToSmiles(mol)
                f.write(smiles + "\n")
            except:
                pass


def save_smiles_to_txt(smiles_list, filename):
    with open(filename, "w") as file:
        for smiles in smiles_list:
            file.write(smiles + "\n")


def evaluate_and_save_log(generated_mols, args, grammar):
    # calculate validity
    validity = calculate_validity_percentage(generated_mols)

    # calculate uniqueness
    uniqueness = calculate_uniqueness_percentage(generated_mols)

    # calculate diversity
    div = InternalDiversity()
    try:
        diversity = div.get_diversity(generated_mols)
    except:
        # remove mol which cannot be converted into MorganFinger print
        generated_mols_for_diversity = []
        for mol in generated_mols:
            try:
                AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
                generated_mols_for_diversity.append(mol)
            except:
                pass
        diversity = div.get_diversity(generated_mols_for_diversity)
    # calculate chamfer
    # Remove duplicated molecules
    with open(args.training_data, "r") as fr:
        lines = fr.readlines()
        mol_sml = []
        for line in lines:
            if not (line.strip() in mol_sml):
                mol_sml.append(line.strip())

    chamfer_dis_min = calculate_min_chamfer_distance(generated_mols, mol_sml)

    chamfer_dis_avg = calculate_avg_chamfer_distance(generated_mols, mol_sml)

    coverage = calculate_coverage_chamfer_distance(generated_mols, mol_sml)

    # calculate novelty
    novelty = percentage_mols_not_in_smiles(generated_mols, mol_sml)

    # calculate RS

    if args.save_sa_score:

        def convert_mols_to_smiles(mols):
            smiles_list = []
            for mol in mols:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                else:
                    pass
            return smiles_list

        def calculate_average_synth_score(mol_list):
            total_score = 0
            valid_count = 0

            smiles_list = convert_mols_to_smiles(mol_list)

            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol is not None:
                    score = sascorer.calculateScore(mol)
                    total_score += score
                    valid_count += 1

            if valid_count == 0:
                return None
            else:
                return total_score / valid_count

        sa_score = calculate_average_synth_score(generated_mols)

    else:

        rs_score = calculate_rs(generated_mols)

        # calculate membership
        membership_percentage = calculate_membership(generated_mols, args)

        # number of useful total molecules

        num_useful_mols, useful_validity, useful_uniqueness, useful_diversity, useful_chamfer, useful_smi_list = (
            calculate_useful_mol_percentage(generated_mols, mol_sml)
        )

        # number of useful total molecules without considering membership

        (
            num_useful_mols_without_membership,
            useful_validity_without_membership,
            useful_uniqueness_without_membership,
            useful_diversity_without_membership,
            useful_chamfer_without_membership,
            useful_smi_list,
        ) = calculate_useful_mol_percentage(generated_mols, mol_sml, without_membership=True)

    # log the results
    # Create logger
    save_log_path = "log/log-evaluate_{}-{}_grammar_training_{}_expr_name_{}_{}".format(
        args.num_generated_samples, time.strftime("%Y%m%d-%H%M%S"), args.grammar_training, args.expr_name, args.task_name
    )
    create_exp_dir(save_log_path, scripts_to_save=[f for f in os.listdir("./") if f.endswith(".py")])
    logger = create_logger("global_logger", save_log_path + "/log.txt")
    logger.info("args:{}".format(pprint.pformat(args)))
    logger = logging.getLogger("global_logger")

    if args.save_sa_score:
        logger.info(
            f"validity : {validity},  uniqueness: {uniqueness}, novelty: {novelty}, diversity: {diversity}, chamfer_dis_min: {chamfer_dis_min}, chamfer_dis_avg : {chamfer_dis_avg} , coverage: {coverage} ,sa_score : {sa_score}"
        )
    else:

        logger.info(
            f"validity : {validity},  uniqueness: {uniqueness}, novelty: {novelty}, diversity: {diversity}, chamfer_dis_min: {chamfer_dis_min}, chamfer_dis_avg : {chamfer_dis_avg} , coverage: {coverage} ,rs_score: {rs_score}, membership_percentage: {membership_percentage},num_useful_mols: {num_useful_mols}, useful_validity: {useful_validity}, useful_uniqueness: {useful_uniqueness}, useful_diversity: {useful_diversity}, useful_chamfer: {useful_chamfer}"
        )
        logger.info(
            f"num_useful_mols_without_membership : {num_useful_mols_without_membership}, useful_validity_without_membership : {useful_validity_without_membership}, useful_uniqueness_without_membership: {useful_uniqueness_without_membership}, useful_diversity_without_membership : {useful_diversity_without_membership}, useful_chamfer_without_membership : {useful_chamfer_without_membership}"
        )

    if args.save_grammar_sample_smi == True and args.save_sa_score == False:
        save_file_path = save_log_path + "/generated_smi.txt"
        save_smiles_to_file(generated_mols, save_file_path)
        save_file_path = save_log_path + "/generated_smi_useful.txt"
        save_smiles_to_txt(useful_smi_list, save_file_path)


def smiles_to_mols(filename):
    """
    Load SMILES from a file, convert them to RDKit mol objects, and return as a list.

    Parameters:
    filename (str): Path to the text file containing SMILES strings, one per line.

    Returns:
    List[rdkit.Chem.rdchem.Mol]: A list of RDKit mol objects.
    """
    mols = []
    with open(filename, "r") as file:
        for line in file:
            smiles = line.strip()  # Remove any extra whitespace
            try:
                mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to mol
                if mol is not None:  # Check if the conversion was successful
                    mols.append(mol)
            except:
                pass
    return mols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC training")
    parser.add_argument(
        "--training_data", type=str, default="./datasets/isocyanates.txt", help="file name of the training data"
    )
    parser.add_argument("--expr_name", type=str, default="isocyanates", help="expr_name name of the data")
    parser.add_argument("--model_folder", type=str, help="folder name of the trained model")
    parser.add_argument(
        "--num_generated_samples", type=int, default=100, help="number of generated samples to evaluate grammar"
    )

    # grammar training
    parser.add_argument("--grammar_training", action="store_true", default=False, help="train grammar using Marcov model")
    parser.add_argument("--early_stopping", type=int, default=100, help="train grammar using Marcov model")

    parser.add_argument("--add_top_k", type=int, default=0, help="add another top k from q table")
    parser.add_argument(
        "--add_top_k_random",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )

    parser.add_argument(
        "--topk_from_scratch",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )
    parser.add_argument(
        "--random_combination",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )
    parser.add_argument("--top_k_selection", type=int, default=0, help="add another top k from q table")
    parser.add_argument("--random_iter_start", type=int, default=15, help="add another top k from q table")

    parser.add_argument(
        "--top_k_based_on_chosen_ones",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )

    parser.add_argument("--add_top_k_starting_ending", type=int, default=0, help="add another top k from q table")

    parser.add_argument(
        "--final_model",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )
    parser.add_argument("--uniform_smoothing_e", type=float, default=0, help="discount factor")
    parser.add_argument("--a_ending_factor", type=float, default=0.5, help="discount factor")

    parser.add_argument("--task_name", type=str, help="task name")

    parser.add_argument(
        "--save_grammar_sample_smi",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )
    parser.add_argument(
        "--save_grammar_rule_png",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )

    parser.add_argument(
        "--save_sa_score",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )

    parser.add_argument("--generated_smi_file", type=str, default=None, help="task name")

    parser.add_argument(
        "--select_frag",
        action="store_true",
        default=False,
        help="add another top k from q table randomly for baseline comparison",
    )
    parser.add_argument("--select_frag_filename", type=str, default=None, help="task name")

    args = parser.parse_args()
    # load the best model and generate 10k samples
    if args.generated_smi_file != None:
        generated_mols = smiles_to_mols(args.generated_smi_file)
        grammar = None
    else:
        generated_mols, grammar = load_and_generate_mols(args)

    # evaluate generated samples
    # save the result
    if args.save_grammar_rule_png or args.select_frag:
        pass
    else:
        evaluate_and_save_log(generated_mols, args, grammar)
