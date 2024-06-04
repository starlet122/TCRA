#!/usr/bin/python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import dgl
import dgl.function as fn 
import utils
from utils import get_param
from decoder import ConvE, DistMult, TransE
import logging
import torch.nn.functional as F
from Transformer_Encoder import Rule_Transformer



class RuGNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']

        self.ent_emb = get_param(self.n_ent, h_dim)
        self.ent_emb1 = get_param(self.n_ent, h_dim)

        # gnn layer
        self.kg_n_layer = self.cfg.kg_layer

        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2 + 1, h_dim) for _ in range(self.kg_n_layer)])
        self.r_rel_emb = get_param(self.n_rel * 2 + 1, h_dim)

        # relation SE layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # entity SE layer
        self.node_layers = nn.ModuleList([NodeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # triple SE layer
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])
        # rule sequence encoding layer
        self.rule_layer = RuleLayer(h_dim)
        # global node SE layer
        self.global_layer = GlobalLayer(h_dim)

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim) 
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2 + 1, h_dim)
        if(self.cfg.decoder=='TransE'):
            self.predictor=TransE()
        elif(self.cfg.decoder=='DistMult'):
            self.predictor=DistMult()
        else:
            self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)


        self.bce = nn.BCELoss()

        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.act = nn.Tanh()

    def forward(self, h_id, r_id, kg, rules, rules_mask, IM, Ht):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        ent_emb, rel_emb = self.aggragate_emb(kg, rules, rules_mask, IM, Ht)
        # print(ent_emb)
        # print(rel_emb)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        
        score = self.predictor(head, rel, ent_emb)
        return score


    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss
    
    def aggragate_emb(self, kg, rules, rules_mask, IM, Ht):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        rel_emb_list = []
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb
            rel_emb_list.append(rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        
        rule_rel_emb = self.rule_layer(rules, rules_mask, IM, self.r_rel_emb)
        g_ent_emb = self.global_layer(Ht, self.ent_emb1)

        rel_emb = pred_rel_emb[:-1,:] + rule_rel_emb
        # rel_emb = pred_rel_emb[:-1,:]
        ent_emb = ent_emb + g_ent_emb

        return ent_emb, rel_emb
    '''
    def set_rules(self, input):
        # input: [rule_id, rule_head, rule_body]
        self.num_rules = len(input)

        # rule_body's max length
        self.max_length = max([len(rule[2:]) for rule in input])

        
        self.relation2rules = [[] for r in range(self.n_rel*2)]
        for rule in input:
            relation = rule[1]    
            self.relation2rules[relation].append([rule[0], (rule[1], rule[2:])]) 
    '''   
        

class RuleLayer(nn.Module):   # rule sequence encoding layer
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.padding_index = self.n_rel * 2
        self.rule_hidden_dim = self.cfg.rule_hidden_dim
        self.rule_dim = self.cfg.rule_dim
        self.rule_len = self.cfg.rule_len
        self.num_head = 2
        self.num_encoder = self.cfg.num_encoder
        self.comp_op = self.cfg.comp_op
        self.h_dim = h_dim
        assert self.comp_op in ['add', 'mul']

        self.fc_R = nn.Linear(self.rule_dim, h_dim) 
        self.fc1 = nn.Linear(h_dim*self.rule_len, self.rule_dim)

        # rule transformer
        self.rule_transformer = Rule_Transformer(h_dim, self.rule_hidden_dim, self.rule_dim, self.rule_len, self.num_head, self.num_encoder, self.cfg.rule_drop)
       
    def forward(self, rules, rules_mask, IM, rel_emb):
        # rules: [rule_id, rule_head, rule_body]  torch.Size([rules_num, 5]) 
        # rules_mask: torch.Size([rules_num, 3])
        # IM: [rules_num, n_rel*2]
        # rule body
        rule_body = rules[:,2:]   # torch.Size([rules_num, 3]) 
        body_embs = rel_emb[rule_body]   # torch.Size([rules_num, 3, h_dim])
        # rule head
        rule_head = rules[:,1]   # torch.Size([rules_num]) 
        head_embs = rel_emb[rule_head]   # torch.Size([rules_num, h_dim])
        head_embs = head_embs.unsqueeze(1) # torch.Size([rules_num, 1, h_dim])
        
        # rule body transformer
        rules_mask = rules_mask.unsqueeze(1)  # torch.Size([rules_num, 1, 3])
        body_embs = self.rule_transformer(body_embs, head_embs, rules_mask)
        # [rules_num, h_dim]

        '''
        # print(body_embs.shape)
        body_embs = body_embs.view(-1, self.h_dim*self.rule_len)
        # print(body_embs.shape)
        # out.shape=[rules_num, self.h_dim*3]
        body_embs = self.fc1(body_embs)
        # print(body_embs.shape)
        '''
        
        dva_i = 1 / IM.sum(0) 
        dva_i[torch.isinf(dva_i)] = 0.0
        IM = IM.transpose(0,1)  # torch.Size([n_rel*2, rules_num])
        pred_rel_emb = IM @ body_embs * dva_i[:, None]
        #pred_rel_emb = torch.mm(IM, body_embs) * dva_i.unsqueeze(-1)    
        # torch.Size([n_rel*2, h_dim])
        pred_rel_emb = F.relu(self.fc_R(pred_rel_emb))
        return pred_rel_emb

class GlobalLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.s_dim = self.cfg.s_dim
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        
        self.s_w = get_param(h_dim, self.s_dim)
        self.fc_R = nn.Linear(self.s_dim, h_dim) 

    def forward(self, Ht, ent_emb):

        De_i = 1 / Ht.sum(0)
        Dv_i = 1 / Ht.sum(1)  
        De_i[torch.isinf(De_i)] = 0.0
        Dv_i[torch.isinf(Dv_i)] = 0.0
        set_embs = (Ht.t() @ ent_emb) * De_i[:, None] @ self.s_w
        gemb = (Ht @ set_embs) * Dv_i[:, None]
        gemb = F.relu(self.fc_R(gemb))
        return gemb

class CompLayer(nn.Module):   # triple SE layer
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.size(0)
        # assert rel_emb.size(0) == 2 * self.n_rel + 1

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            # neihgbor entity and relation composition
            if self.cfg.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.cfg.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):   # # entity SE layer
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.size(0)

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # attention
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):   # relation SE layer
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = utils.get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.size(0)
        # assert rel_emb.size(0) == 2 * self.n_rel + 1 

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # attention
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
