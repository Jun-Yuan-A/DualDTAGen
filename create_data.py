import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import json, pickle
import torch
import esm
import math
import networkx as nx
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected, remove_self_loops, coalesce
from utils import * 


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]
    
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + 
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + 
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + 
                    [atom.GetIsAromatic()] + 
                    [atom.IsInRing()] 
                    )

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(bond_feats)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edge_feats = bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'edge_feats': edge_feats}))
        
    g = nx.Graph()
    g.add_edges_from(edges)
    g = g.to_directed()
    edge_index = []
    edge_feats = []
    for e1, e2, feats in g.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(feats['edge_feats'])
        
    return c_size, features, edge_index, edge_feats



def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

def seq_feature(pro_seq):    
    if 'U' in pro_seq or 'B' in pro_seq:
        pass 
    pro_seq = pro_seq.replace('U','X').replace('B','X')
    pro_property = np.zeros((len(pro_seq), 12)) 
    for i in range(len(pro_seq)):
        pro_property[i,] = residue_features(pro_seq[i])
    return pro_property 

def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_map_proba[row, col].float()
    
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1],(torch.arange(num_residues)+1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:],(torch.arange(num_residues)-1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1],dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2],(torch.arange(num_residues)+2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:],(torch.arange(num_residues)-2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2],dim=-1)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,fill_value=1)
    
    return edge_index, edge_weight

def esm_extract(model, batch_converter, seq, layer=33, approach='mean', dim=1280):
    pro_id = 'A'
    if len(seq) <= 1000: 
        data = []
        data.append((pro_id, seq))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=True)

        contact_prob_map = results["contacts"][0].cpu() 
        token_representation = results["representations"][layer][0, 1: len(seq) + 1].cpu() 
        
    else:
        contact_prob_map = torch.zeros((len(seq), len(seq)))
        token_representation = torch.zeros((len(seq), dim))
        interval = 256 
        window_size = 1000
        
        steps = math.ceil((len(seq) - window_size) / interval) + 1
        
        count_map = torch.zeros((len(seq), len(seq)))
        token_count = torch.zeros((len(seq), 1))
        
        for s in range(steps):
            start = s * interval
            end = min(start + window_size, len(seq))
            sub_seq_len = end - start
            if sub_seq_len < 5: break 

            temp_seq = seq[start:end]
            temp_data = [(pro_id, temp_seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[layer], return_contacts=True)
            
            sub_contact = results["contacts"][0].cpu()
            contact_prob_map[start:end, start:end] += sub_contact
            count_map[start:end, start:end] += 1
            
            sub_repr = results["representations"][layer][0, 1: len(temp_seq) + 1].cpu()
            token_representation[start:end] += sub_repr
            token_count[start:end] += 1
            
            if end == len(seq): break
            
        contact_prob_map = torch.div(contact_prob_map, count_map + 1e-6) 
        token_representation = torch.div(token_representation, token_count + 1e-6)

    return token_representation, contact_prob_map

def generate_psichic_graph(seq, model, batch_converter):
    token_repr, contact_proba = esm_extract(model, batch_converter, seq, layer=33, dim=1280)
    phys_feat_np = seq_feature(seq) 
    phys_feat = torch.from_numpy(phys_feat_np).float()
    node_x = torch.cat([token_repr, phys_feat], dim=1)
    edge_index, edge_weight = contact_map(contact_proba)
    return Data(x=node_x, edge_index=edge_index, edge_attr=edge_weight)




print("Loading ESM-2 Model...")
esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model.eval()
if torch.cuda.is_available():
    esm_model = esm_model.cuda()
esm_batch_converter = esm_alphabet.get_batch_converter()


dir_path = 'data'
processed_dir = 'data/processed'


if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

datasets = ['davis', 'kiba', 'bindingdb']
scenarios = ['cold_drug', 'cold_target'] 

for dataset in datasets:
    for scenario in scenarios:

        base_name = f"{dataset}_{scenario}"
        

        processed_data_file_train = f'{processed_dir}/{base_name}_train.pt'
        processed_data_file_test = f'{processed_dir}/{base_name}_test.pt'
        tokenizer_file = f'{dir_path}/{base_name}_tokenizer.pkl'
        
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print(f"========== Processing {base_name} ==========")
            
       
            csv_train_path = f'{dir_path}/{base_name}_train.csv'
            csv_test_path = f'{dir_path}/{base_name}_test.csv'
            
            if not os.path.exists(csv_train_path):
                print(f"[Warning] 文件 {csv_train_path} 不存在，跳过该数据集。")
                continue
            
            df_train = pd.read_csv(csv_train_path)
            df_test = pd.read_csv(csv_test_path)

        
            all_smiles = set(df_train['compound_iso_smiles']).union(set(df_test['compound_iso_smiles']))
            tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))
            with open(tokenizer_file, 'wb') as file:
                pickle.dump(tokenizer, file)        
            
        
            compound_iso_smiles = set(list(df_train['compound_iso_smiles']) + list(df_test['compound_iso_smiles']))
            smile_graph = {}
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g

     
            all_prots = set(list(df_train['target_sequence']) + list(df_test['target_sequence']))
            prot_graph = {}
            print(f"Generating Protein Graphs for {len(all_prots)} unique sequences in {base_name}...")
            
            for i, prot in enumerate(tqdm(all_prots)):
                try:
                    pg = generate_psichic_graph(prot, esm_model, esm_batch_converter)
                    prot_graph[prot] = pg
                except Exception as e:
                    print(f"Error processing protein index {i}: {e}")
                    prot_graph[prot] = Data(x=torch.zeros(len(prot), 1292), edge_index=torch.zeros(2,0).long())

       
            train_drugs = np.asarray(list(df_train['compound_iso_smiles']))
            train_MTS = np.asarray(list(df_train['target_smiles']))
            train_Y = np.asarray(list(df_train['affinity']))
            train_prots_seq = list(df_train['target_sequence'])
            
            train_prot_graphs = [prot_graph[p] for p in train_prots_seq] 
            train_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in train_MTS]

     
            test_drugs = np.asarray(list(df_test['compound_iso_smiles']))
            test_MTS = np.asarray(list(df_test['target_smiles']))
            test_Y = np.asarray(list(df_test['affinity']))
            test_prots_seq = list(df_test['target_sequence'])
            
            test_prot_graphs = [prot_graph[p] for p in test_prots_seq]
            test_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in test_MTS]

         
            print(f'saving {base_name}_train.pt to {processed_dir} ...')
            train_data = TestbedDataset(root='data', dataset=f'{base_name}_train', 
                                        xd=train_drugs, xdt=train_XD, 
                                        xt=train_prot_graphs, 
                                        y=train_Y, smile_graph=smile_graph)
            
            print(f'saving {base_name}_test.pt to {processed_dir} ...')
            test_data = TestbedDataset(root='data', dataset=f'{base_name}_test', 
                                       xd=test_drugs, xdt=test_XD, 
                                       xt=test_prot_graphs, 
                                       y=test_Y, smile_graph=smile_graph)       
        else:
            print(f"{processed_data_file_train} already exists, skipping...")

print("All cold start datasets processing finished!")