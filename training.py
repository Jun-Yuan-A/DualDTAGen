import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader
from utils import *
from model import DualDTAGen
from FetterGrad import FetterGrad

from tqdm import tqdm
import sys, os
import time
import pickle
import random


seed = 4221
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    generator = torch.Generator('cuda').manual_seed(seed)
else:
    generator = torch.Generator().manual_seed(seed)



def train(model, device, train_loader, optimizer, mse_f, epoch, FLAGS):
    model.train()
    

    loss = torch.tensor(0.0)
    mse_loss = torch.tensor(0.0)
    kl_loss = torch.tensor(0.0)
    lm_loss = torch.tensor(0.0)
    
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for i, data in enumerate(t):
            data = data.to(device)
            optimizer.zero_grad()
            
          
            Pridection, prediction_scores, lm_loss, kl_loss = model(data)

         
            mse_loss = mse_f(Pridection, data.y.view(-1, 1).float())

      
            train_ci = get_cindex(Pridection.cpu().detach().numpy(), data.y.view(-1, 1).float().cpu().detach().numpy())

         
            loss = kl_loss * 0.001 + mse_loss + lm_loss
            
       
            losses = [loss, mse_loss] 
            optimizer.ft_backward(losses)
            optimizer.step()
            
        
            t.set_postfix(MSE=mse_loss.item(), Train_cindex=train_ci, KL=kl_loss.item(), LM=lm_loss.item())

   
    msg = f"Epoch {epoch+1}, total loss={loss.item()}, MSE={mse_loss.item()}, KL_loss={kl_loss.item()}, LM={lm_loss.item()}"
    logging(msg, FLAGS)
    
    return model

def test(model, device, test_loader, dataset, FLAGS):
  
    print('Testing on {} samples...'.format(len(test_loader.dataset)))
    model.eval()
    
    total_true = []
    total_predict = []
    total_loss = 0 

    if dataset == "kiba":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]  

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            
       
            Pridection, prediction_scores, lm_loss, kl_loss = model(data)

        
            total_true.append(data.y.view(-1, 1).cpu())
            total_predict.append(Pridection.cpu())
            
         
            loss = lm_loss + kl_loss
            total_loss += loss.item() * data.num_graphs


    total_true = torch.cat(total_true, 0)
    total_predict = torch.cat(total_predict, 0)
    
    G = total_true.numpy().flatten()
    P = total_predict.numpy().flatten()
    

    mse_loss = mse(G, P)
    test_ci = get_cindex(G, P)      
    rm2 = get_rm2(G, P)   
    
    auc_values = []
    for t in thresholds:
        auc = get_aupr(np.int32(G > t), P)
        auc_values.append(auc) 
        
    return total_loss, mse_loss, test_ci, rm2, auc_values, G, P

def experiment(FLAGS, dataset, device):
    logging('Starting program', FLAGS)

    # Hyperparameters
    BATCH_SIZE = 32 
    LR = 0.0002
    NUM_EPOCHS = 500

    # Print hyperparameters
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")

    msg = f"Dataset {dataset}, Device {device}, batch size {BATCH_SIZE}, learning rate {LR}, epochs {NUM_EPOCHS}"
    logging(msg, FLAGS)

    # Load tokenizer
    tokenizer_path = f'data/{dataset}_tokenizer.pkl'
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run create_data.py first.")
        
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load processed data
    processed_data_file_train = f"data/processed/{dataset}_train.pt"
    processed_data_file_test = f"data/processed/{dataset}_test.pt"
    
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
        return
    else:
        print("Loading data...")
        train_data = TestbedDataset(root="data", dataset=f"{dataset}_train")
        test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")

     
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, follow_batch=['target_x'])
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, follow_batch=['target_x'])
        
        # Initialize model
        print("Initializing model...")
        model = DualDTAGen(tokenizer).to(device)
        
        # Optimizer
        optimizer = FetterGrad(optim.Adam(model.parameters(), lr=LR))
        mse_f = nn.MSELoss()

        # Train loop
        best_mse = float('inf')  
        best_epoch = -1
        
        for epoch in range(NUM_EPOCHS):
         
            model = train(model, device, train_loader, optimizer, mse_f, epoch, FLAGS)

      
            if (epoch + 1) % 20 == 0: 
                total_loss, mse_loss, test_ci, rm2, auc_values, G, P = test(model, device, test_loader, dataset, FLAGS)
                
                print(f"Test Results - Epoch {epoch+1}:")
                print(f"MSE: {mse_loss:.5f}")
                print(f"CI:  {test_ci:.5f}")
                print(f"RM2: {rm2:.5f}")
                
                if mse_loss < best_mse:
                    best_mse = mse_loss
                    best_epoch = epoch + 1
                    filename = f"saved_models/dualdtagen_model_{dataset}.pth"
                    torch.save(model.state_dict(), filename)
                    
                    # Save best predictions
                    folder_path = "Affinities/"
                    np.savetxt(folder_path + f"estimated_labels_{dataset}.txt", P)
                    np.savetxt(folder_path + f"true_labels_{dataset}.txt", G)
                    print(f'Model saved (Best MSE: {best_mse:.5f})')
                
                print(f"AUCs: {', '.join([f'{auc:.4f}' for auc in auc_values])}")

        logging(f'Program finished. Best MSE: {best_mse} at epoch {best_epoch}', FLAGS)

if __name__ == "__main__":
    datasets = ['davis', 'kiba', 'bindingdb']
    dataset_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0 
    dataset = datasets[dataset_idx]

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_arg = "cuda:" + str(int(sys.argv[2])) if len(sys.argv) > 2 else default_device
    device = torch.device(device_arg if torch.cuda.is_available() else "cpu")

    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = f'dataset_{dataset}_{int(time.time())}'

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    os.makedirs('Affinities', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    experiment(FLAGS, dataset, device)