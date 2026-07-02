import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import pickle
from pathlib import Path
import pandas as pd
from rdkit import RDLogger, Chem


from torch_geometric.loader import DataLoader 

from tqdm.auto import tqdm
from model import DualDTAGen
from utils import *
import torch

RDLogger.DisableLog('rdApp.*')


def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model = DualDTAGen(tokenizer)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states, strict=False))

    return model, tokenizer


def format_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['kiba', 'bindingdb'], help='the dataset name (kiba or bindingdb)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use (cpu or cuda)')
    args = parser.parse_args()

    dataset = args.dataset
    device = args.device

    config = {
        'input_path': f'data/{dataset}_test.csv',
        'model_path': f'saved_models/dualdtagen_model_{dataset}.pth',
        'tokenizer_path': f'data/{dataset}_tokenizer.pkl',
        'n_mol': 40000,
        'filter': True,
        'batch_size': 1,
        'seed': -1
    }

    # Load the input CSV
    input_df = pd.read_csv(config['input_path'])
    input_df['Generated_SMILES'] = None

    # Load dataset and model
    test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model, tokenizer = load_model(config['model_path'], config['tokenizer_path'])

    model.eval()
    model.to(device)

    # Generate SMILES
    for i, data in enumerate(tqdm(test_loader)):
      
        data = data.to(device) 
        
        generated = tokenizer.get_text(model.generate(data))
        generated = generated[:config['n_mol']]

        if config['filter']:
            generated = [format_smiles(smi) for smi in generated]
            generated = [smi for smi in generated if smi]

        input_df.loc[i, 'Generated_SMILES'] = generated[0] if generated else None

    output_path = Path(f"{dataset}_generated.csv")
    input_df.to_csv(output_path, index=False)
    print(f'Generation complete. Output saved to {output_path}')