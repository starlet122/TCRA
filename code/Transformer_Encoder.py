import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
import utils

class Positional_Encoding(nn.Module):
    '''
    params: embed-->word embedding dim      pad_size-->max_sequence_lenght
    Input: x
    Output: x + position_encoder
        '''
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert len(x.shape)==3
        out = x + nn.Parameter(self.pe.unsqueeze(0), requires_grad=False).cuda()
        out = self.dropout(out)
        return out
    
class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden_dim dim      num_head
    '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    
        self.dim_head = dim_model // self.num_head   
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, y, z, mask=None):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(y)
        V = self.fc_V(z)
        # reshape to batch*head*sequence_length*(embedding_dim//head)
        Q = Q.view(batch_size ,  -1, self.num_head,   self.dim_head).transpose(1,2)
        K = K.view(batch_size ,  -1,  self.num_head,  self.dim_head).transpose(1,2)
        V = V.view(batch_size ,  -1, self.num_head,   self.dim_head).transpose(1,2)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale,mask=mask) 
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.dim_head * self.num_head) 
        out = self.fc(context)   
        out = self.dropout(out)
        out = out + x      
        out = self.layer_norm(out)
        return out
    
class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None,mask=None):
        attention = torch.matmul(Q, K.transpose(-2,-1))  # Q*K^T
        if scale:#attention.shape=[bz,h,pad_size,pad_size]
            attention = attention * scale
        if mask is not None:  # TODO change this#mask.shape=[bz,1,pad_size]
            mask=mask.unsqueeze(1)#mask.shape=[bz,1,1,pad_size]
            attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context
    
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden_dim, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out
    
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden_dim, dropout)

    def forward(self, Q, K, V, mask=None):
        out = self.attention(Q, K, V, mask=mask)
        out = self.feed_forward(out)
        return out


class Rule_Transformer(nn.Module):
    def __init__(self, dim_model, hidden_dim, out_dim, rule_len, num_head, num_encoder, dropout=0.0):
        super(Rule_Transformer, self).__init__()
        self.dim_model = dim_model
        self.postion_embedding = Positional_Encoding(dim_model, rule_len, dropout)
        self.encoder = Encoder(dim_model, num_head, hidden_dim, dropout)
        self.encoders = nn.ModuleList([Encoder(dim_model, num_head, hidden_dim, dropout) for _ in range(num_encoder)])
        self.fc1 = nn.Linear(dim_model, out_dim)


    def forward(self, x, tgt_rel, mask=None):
        # out = self.postion_embedding(x)
        out = x
        for encoder in self.encoders[:-1]:
            out = encoder(out, out, out, mask)
        # out.shape=[rules_num, pad_size, h_dim]

        out = self.encoders[-1](tgt_rel,out,out,mask)
        # out.shape=[rules_num, 1, h_dim]
        out = out.squeeze(1)
        # out.shape=[rules_num, h_dim]
        out = self.fc1(out)
        # out.shape=[rules_num, out_dim]
        return out