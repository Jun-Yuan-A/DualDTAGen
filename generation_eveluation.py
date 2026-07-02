import pandas as pd
from rdkit import Chem
import argparse
import os


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is chemically valid."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


def evaluate_smiles(smiles_list, reference_set=None):
    """Evaluate validity, uniqueness, and novelty of SMILES."""
    valid = [s for s in smiles_list if is_valid_smiles(s)]
    unique = set(valid)
    novel = [s for s in unique if s not in reference_set] if reference_set else list(unique)

    return {
        "validity_ratio": len(valid) / len(smiles_list) if smiles_list else 0,
        "uniqueness_ratio": len(unique) / len(valid) if valid else 0,
        "novelty_ratio": len(novel) / len(unique) if unique else 0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bindingdb', help='Dataset prefix (default: "bindingdb")')
    args = parser.parse_args()

    file_path = f"{args.dataset}_generated.csv"
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    df = pd.read_csv(file_path)
    if 'Generated_SMILES' not in df.columns:
        print("Error: Column 'Generated_SMILES' not found in the dataset.")
        return

    generated = df['Generated_SMILES'].dropna().tolist()
    reference_set = set(df['target_smiles'].dropna()) if 'target_smiles' in df.columns else None

    results = evaluate_smiles(generated, reference_set)

    print(f"Validity Ratio   : {results['validity_ratio']:.2f}")
    print(f"Uniqueness Ratio : {results['uniqueness_ratio']:.2f}")
    print(f"Novelty Ratio    : {results['novelty_ratio']:.2f}")


if __name__ == "__main__":
    main()
