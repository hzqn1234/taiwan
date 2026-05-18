import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math
from utils import args

class Amodel(nn.Module):
    def __init__(self, series_dim, feature_dim, target_num, hidden_num, hidden_dim, drop_rate=0.5, use_series_oof=False):
        super(Amodel, self).__init__()
        self.use_series_oof = use_series_oof
        hidden_feature_dropout = 0.005
        self.input_series_block = nn.Sequential(
                                         # nn.LayerNorm(series_dim)
                                        nn.Linear(series_dim, hidden_dim)
                                        ,nn.LayerNorm(hidden_dim)
                                        )
        self.input_feature_block = nn.Sequential(
                                        nn.Linear(feature_dim, hidden_dim)
                                        ,nn.BatchNorm1d(hidden_dim)
                                        ,nn.Dropout(hidden_feature_dropout) #drop_rate
                                        ,nn.LeakyReLU()
                                        )
        # self.gru_series = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.gru_series = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
                                                    d_model         = hidden_dim, 
                                                    nhead           = 2, 
                                                    dim_feedforward = 4, 
                                                    dropout         = 0.005 #0.001
                                                    # dropout         = 0
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # decoder_layer = nn.TransformerDecoderLayer(
        #                                             d_model         = hidden_dim, 
        #                                             nhead           = 2, 
        #                                             dim_feedforward = 128, 
        #                                             dropout         = 0.1
        #                                             )
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.hidden_feature_block = []
        # for h in range(hidden_num-1):
        for h in range(2):
            self.hidden_feature_block.extend([
                                     nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.BatchNorm1d(hidden_dim)
                                     ,nn.Dropout(hidden_feature_dropout) #drop_rate)
                                     ,nn.LeakyReLU()
                                     ])
        self.hidden_feature_block = nn.Sequential(*self.hidden_feature_block)

        self.output_block = nn.Sequential(
                                        # nn.BatchNorm1d(2*hidden_dim if use_series_oof else 1*hidden_dim)  
                                        nn.BatchNorm1d(3*hidden_dim if use_series_oof else 2*hidden_dim)                             
                                        ,nn.Linear(3*hidden_dim if use_series_oof else 2*hidden_dim, 1*hidden_dim)
                                          # ,nn.Linear(2*hidden_dim if use_series_oof else 1*hidden_dim, 1*hidden_dim)
                                          # nn.Linear(1*hidden_dim if use_series_oof else 1*hidden_dim, 1*hidden_dim)
                                         # ,nn.BatchNorm1d(hidden_dim)
                                        ,nn.Dropout(0.025) 
                                        ,nn.LeakyReLU()

                                          # ,nn.BatchNorm1d(hidden_dim)
                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.LeakyReLU()
                                         
                                         ,nn.Linear(1*hidden_dim, target_num)
                                         ,nn.Sigmoid()
                                         )

    def batch_gru(self,series,mask):
        node_num = mask.sum(dim=-1).detach().cpu()
        pack = nn.utils.rnn.pack_padded_sequence(series, node_num, batch_first=True, enforce_sorted=False)
        message,hidden = self.gru_series(pack)
        pooling_feature = []

        for i,n in enumerate(node_num.numpy()):
            n = int(n)
            bi = 0

            si = message.unsorted_indices[i]
            for k in range(n):

                if k == n-1:
                    sample_feature = message.data[bi+si]
                bi = bi + message.batch_sizes[k]

            pooling_feature.append(sample_feature)
        return torch.stack(pooling_feature,0)

    def transformer_pooling(self, transfomer_message, mask):
        node_num = mask.sum(dim=-1).detach().cpu().tolist()
        node_num_int = [int(a) for a in node_num]
        pooling_feature = []
        
        for i in range(len(node_num_int)):
            sample_feature = transfomer_message[:,i,:][node_num_int[i]-1]
            
            # # temporal pooling
            # if node_num_int[i] >=3:
            #     sample_feature = transfomer_message[:,i,:][node_num_int[i]-3:node_num_int[i]].mean(0)
            # else:
            #     sample_feature = transfomer_message[:,i,:][:node_num_int[i]].mean(0)
            pooling_feature.append(sample_feature)
        
        return torch.stack(pooling_feature,0)
    
    def forward(self, data):
        x1 = self.input_series_block(data['batch_series'])
        # tgt = torch.zeros(x1.shape).permute(1, 0, 2).cuda()
        # tgt = self.input_series_block(torch.zeros(data['batch_series'].shape).cuda()).permute(1, 0, 2)

        # x1 = self.batch_gru(x1,data['batch_mask'])
        out = x1
        # print('input (n_samples, n_length, n_channel)', out.shape)
        
        out = out.permute(1, 0, 2)
        # print('transpose (n_length, n_samples, n_channel)', out.shape)

        out = self.transformer_encoder(out)
        # print('transformer_encoder', out.shape)
        
        # tgt = torch.zeros(x1.shape).cuda()
        # out = self.transformer_decoder(tgt,out)
        
        # out = out[-1]
        # out = self.transformer_pooling(out,data['batch_mask'])
        # out = out.mean(0)
        
        out = out.permute(1, 0, 2).cuda()
        out = self.batch_gru(out,data['batch_mask'])
        
        x1 = out
        
        
        if self.use_series_oof:
            x2 = self.input_feature_block(data['batch_feature'])
            x2 = self.hidden_feature_block(x2)
            x = torch.cat([x1,x2],axis=1)
            y = self.output_block(x)
            # y = self.output_block(x2)
        else:
            y = self.output_block(x1)
        return y
