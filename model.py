import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv, global_max_pool as gmp, global_mean_pool as gap
from typing import Optional, Dict
import math
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from utils import Tokenizer
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x 


class Encoder(torch.nn.Module):
    def __init__(self, Drug_Features, dropout, Final_dim, Protein_Dim=128):
        super(Encoder, self).__init__()
        self.hidden_dim = 376
        
        # Drug Graph Layers
        self.GraphConv1 = GCNConv(Drug_Features, Drug_Features * 2)
        self.GraphConv2 = GCNConv(Drug_Features * 2, Drug_Features * 3)
        self.GraphConv3 = GCNConv(Drug_Features * 3, Drug_Features * 4)
        
  
        self.cond = nn.Linear(Protein_Dim, self.hidden_dim) 
        
        self.mean = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        self.var = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.Drug_FCs = nn.Sequential(
            nn.Linear(Drug_Features * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, Final_dim)
        )
        self.Relu_activation = nn.ReLU()
        self.pp_seg_encoding = nn.Parameter(torch.randn(376))
    
    def reparameterize(self, z_mean, logvar, batch, con, a):
        # Compute KL
        z_log_var = -torch.abs(logvar)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / 64
        
        epsilon = torch.randn_like(z_mean).to(z_mean.device)
        z_ = z_mean + torch.exp(z_log_var / 2) * epsilon
        
    
        con_embedding = self.cond(con)
        
        # Add conditioning and auxiliary factor
        z_ = z_ + con_embedding + a
        return z_, kl_loss

    def process_p(self, node_features, num_nodes, batch_size):
        # Convert graph features to sequence
        d_node_features = pad_sequence(torch.split(node_features, num_nodes.tolist()), batch_first=False, padding_value=-999)
        padded_sequence = d_node_features.new_ones((d_node_features.shape[0], 
                                                    d_node_features.shape[1], 
                                                    d_node_features.shape[2])) * -999
        padded_sequence[:d_node_features.shape[0], :, :] = d_node_features
        d_node_features = padded_sequence
        padding_mask = (d_node_features[:, :, 0].T == -999).bool()
        padded_sequence_with_encoding = d_node_features + self.pp_seg_encoding
        return padded_sequence_with_encoding, padding_mask

    def forward(self, data, con):
       x, edge_index, batch, num_nodes, affinity = data.x, data.edge_index, data.batch, data.c_size, data.y
       a = affinity.view(-1, 1)
       
       # Drug GCN
       x = self.GraphConv1(x, edge_index)
       x = self.Relu_activation(x)
       x = self.GraphConv2(x, edge_index)
       x = self.Relu_activation(x)
       x = self.GraphConv3(x, edge_index)
       PMVO_pre = x # Last layer features
       x = self.Relu_activation(x)
       
       # For VAE
       d_sequence, Mask = self.process_p(x, num_nodes, batch)
       mu = self.mean(d_sequence)
       logvar = self.var(d_sequence)
       
       # Conditioning
       AMVO, kl_loss = self.reparameterize(mu, logvar, batch, con, a)
       
       # For Prediction (Pooling)
       x2 = gmp(x, batch)
       PMVO = self.Drug_FCs(x2)
       return d_sequence, AMVO, Mask, PMVO, kl_loss


class Decoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        for layer in self.layer:
            x = layer(x, mem,
                      self_attn_mask=x_mask, self_attn_padding_mask=x_padding_mask,
                      encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

    @torch.jit.export
    def forward_one(self, x, mem, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]], mem_padding_mask: torch.BoolTensor = None):
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state, encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x


class ProteinGNN(nn.Module):
    def __init__(self, num_features_xt, output_dim, dropout=0.2):
        super(ProteinGNN, self).__init__()
        
        self.conv1 = GCNConv(num_features_xt, num_features_xt)
        self.conv2 = GCNConv(num_features_xt, 512)
        self.conv3 = GCNConv(512, output_dim * 2) 
        
        self.fc = nn.Sequential(
            nn.Linear(output_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim) 
        )
        self.relu = nn.ReLU()


    def forward(self, target_x, target_edge_index, target_batch, target_edge_weight=None):
        # GCN Layers
        x = self.conv1(target_x, target_edge_index, edge_weight=target_edge_weight)
        x = self.relu(x)
        x = self.conv2(x, target_edge_index, edge_weight=target_edge_weight)
        x = self.relu(x)
        x = self.conv3(x, target_edge_index, edge_weight=target_edge_weight)
        x = self.relu(x)
        
      
        x = gmp(x, target_batch)
        
        # FC
        xt = self.fc(x)
        return xt


class FC(torch.nn.Module):
    def __init__(self, output_dim, n_output, dropout):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(output_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output)
        )

    def forward(self, Drug_Features, Protein_Features):
        Combined = torch.cat((Drug_Features, Protein_Features), 1)
        Pridection = self.FC_layers(Combined)
        return Pridection


class DualDTAGen(torch.nn.Module):
    def __init__(self, tokenizer):
        super(DualDTAGen, self).__init__()
        self.hidden_dim = 376
        self.max_len = 128
        self.node_feature = 94 
       
        self.drug_features_dim = 94
        
        self.output_dim = 128
        self.ff_dim = 1024
        self.heads = 8
        self.layers = 8
        self.encoder_dropout = 0.2
        self.dropout = 0.3
        

        self.protein_features_dim = 1292 # 1280 (ESM) + 12 (Physicochemical)

        # 1. Protein Module: Replaced CNN with GNN
        self.protein_gnn = ProteinGNN(num_features_xt=self.protein_features_dim, 
                                      output_dim=self.output_dim, 
                                      dropout=self.dropout)

        # 2. Encoder: Passed Protein_Dim
        self.encoder = Encoder(Drug_Features=self.drug_features_dim, 
                               dropout=self.encoder_dropout, 
                               Final_dim=self.output_dim,
                               Protein_Dim=self.output_dim) # Encoder cond dim = 128

        self.decoder = Decoder(dim=self.hidden_dim, ff_dim=self.ff_dim, num_head=self.heads, num_layer=self.layers)
        self.dencoder = TransformerEncoder(dim=self.hidden_dim, ff_dim=self.ff_dim, num_head=self.heads, num_layer=self.layers)
        self.pos_encoding = PositionalEncoding(self.hidden_dim, max_len=256)

        # 3. Fully connected layer
        self.fc = FC(output_dim=self.output_dim, n_output=1, dropout=self.dropout)

        # Learnable parameter for segment encoding
        self.zz_seg_encoding = nn.Parameter(torch.randn(self.hidden_dim))

        # Word prediction layers
        vocab_size = len(tokenizer)
        self.word_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, vocab_size)
        )
        torch.nn.init.zeros_(self.word_pred[3].bias)

        self.vocab_size = vocab_size
        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.word_embed = nn.Embedding(vocab_size, self.hidden_dim)
        self.unk_index = Tokenizer.SPECIAL_TOKENS.index('<unk>')

        self.expand = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Rearrange('batch_size h -> 1 batch_size h')
        )

    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = z
        zzs = zz + self.zz_seg_encoding
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1) 
        zzz = torch.cat((vvs, zzs), dim=0) 
        zzz = self.dencoder(zzz, full_mask)
        return zzz, full_mask

    def sample(self, batch_size, device):
        z = torch.randn(1, self.hidden_dim).to(device)
        return z

    def forward(self, data):
      
        if hasattr(data, 'target_x_batch'):
             target_batch = data.target_x_batch
        else:
             
             target_batch = torch.zeros(data.target_x.size(0), dtype=torch.long, device=data.target_x.device)
        
        edge_weight = data.target_edge_attr if hasattr(data, 'target_edge_attr') else None
        
        Protein_vector = self.protein_gnn(data.target_x, data.target_edge_index, target_batch, edge_weight)
        
    
        con = Protein_vector 
        
        # 2. Encode Drug Graph (with Protein Condition)
        vss, AMVO, mask, PMVO, kl_loss = self.encoder(data, con)
        
        # 3. Expansion and Fusion for Generation
        zzz, encoder_mask = self.expand_then_fusing(AMVO, mask, vss)
        
        # 4. Decoding (Target Sequence Generation)
        targets = data.target_seq
        _, target_length = targets.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool), diagonal=1).to(targets.device)
        target_embed = self.word_embed(targets)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())
        
        output = self.decoder(target_embed, zzz, x_mask=target_mask, mem_padding_mask=encoder_mask).permute(1, 0, 2).contiguous()
        prediction_scores = self.word_pred(output)
        # 5. Loss Calculation
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        batch_size, sequence_length, vocab_size = shifted_prediction_scores.size()
        shifted_prediction_scores = shifted_prediction_scores.view(-1, vocab_size)
        targets = targets.view(-1)
        
        # 6. Affinity Prediction (Drug Feature + Protein Feature)
        Pridection = self.fc(PMVO, Protein_vector)
        
        lm_loss = F.cross_entropy(shifted_prediction_scores, targets, ignore_index=self.pad_value)
        
        return Pridection, prediction_scores, lm_loss, kl_loss

    def _generate(self, zzz, encoder_mask, random_sample, return_score=False):
        batch_size = zzz.shape[1]
        device = zzz.device
        token = torch.full((batch_size, self.max_len), self.pad_value, dtype=torch.long, device=device)
        token[:, 0] = self.sos_value
        text_pos = self.pos_encoding.pe
        text_embed = self.word_embed(token[:, 0]) + text_pos[0]
        text_embed = text_embed.unsqueeze(0)
        
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {})
        scores = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, self.max_len):
            one = self.decoder.forward_one(text_embed, zzz, incremental_state, mem_padding_mask=encoder_mask).squeeze(0)
            l = self.word_pred(one)
            if return_score: scores.append(l)
            
            if random_sample:
                k = torch.multinomial(torch.softmax(l, 1), 1).squeeze(1)
            else:
                k = torch.argmax(l, -1)
            
            token[:, t] = k
            finished |= k == self.eos_value
            if finished.all(): break
            
            text_embed = self.word_embed(k) + text_pos[t]
            text_embed = text_embed.unsqueeze(0)

        predict = token[:, 1:]
        if return_score: return predict, torch.stack(scores, dim=1)
        return predict

    def generate(self, data, random_sample=False, return_z=False):
     
        if hasattr(data, 'target_x_batch'):
             target_batch = data.target_x_batch
        else:
             target_batch = torch.zeros(data.target_x.size(0), dtype=torch.long, device=data.target_x.device)
        
      
        edge_weight = data.target_edge_attr if hasattr(data, 'target_edge_attr') else None
      
        Protein_vector = self.protein_gnn(data.target_x, data.target_edge_index, target_batch, edge_weight)
        con = Protein_vector
        
        vss, AMVO, mask, PMVO, kl_loss = self.encoder(data, con)
        # z = self.sample(data.batch, device=vss.device) 
        # zzz, encoder_mask = self.expand_then_fusing(z, mask, vss)
        
   
        zzz, encoder_mask = self.expand_then_fusing(AMVO, mask, vss)
        predict = self._generate(zzz, encoder_mask, random_sample=random_sample, return_score=False)
        
        if return_z:
            return predict, AMVO.detach().cpu().numpy()
        return predict