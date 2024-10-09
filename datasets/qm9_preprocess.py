import random

import pandas as pd


def process_smiles_data(csv_filename):
    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Select data in the column named "smi"
    smiles_data = df["smi"]

    # Save SMILES data to a text file
    txt_filename = "qm9.txt"
    with open(txt_filename, "w") as file:
        for smi in smiles_data:
            file.write(smi + "\n")

    # Randomly select 100 SMILES and save them to a separate file
    random_100_smiles = random.sample(smiles_data.tolist(), 2000)
    with open("qm9_random_2000_smiles.txt", "w") as file:
        for smi in random_100_smiles:
            file.write(smi + "\n")

    # Randomly select 1000 SMILES and save them to a separate file
    random_1000_smiles = random.sample(smiles_data.tolist(), 5000)
    with open("qm9_random_5000_smiles.txt", "w") as file:
        for smi in random_1000_smiles:
            file.write(smi + "\n")

    print("Processing complete!")


# Example usage:
process_smiles_data("train.csv")
