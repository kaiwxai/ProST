import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import Batch, Data
from ProG.utils import act
import warnings
from deprecated.sphinx import deprecated
import copy
import torch.nn as nn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)
        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p
    

class SubPromptPre(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        super(SubPromptPre, self).__init__()
        self.inner_prune = inner_prune
        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")
        print(self.token_list)

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            # 根据邻接矩阵重新生成索引信息
            edge_index = inner_adj.nonzero().t().contiguous()
            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch
        


class SubPrompt(SubPromptPre):
    def __init__(self, token_dim, token_num, cross_prune=0.3, inner_prune=0.01):
        super(SubPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune
        self.prompt = GPFplusAtt(100, 10)

    def forward(self, graph_batch: Batch):
        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g.x = self.prompt.add(g.x)
            # 插入的操作在这里， 一个token插入到
            cross_dot = torch.mm(g.x, torch.transpose(g.x, 0, 1))

            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            edge_index = torch.cat([g.edge_index, cross_edge_index], dim=1)
            data = Data(x=g.x, edge_index=edge_index, y=g.y)
            re_graph_list.append(data)
        graph_batch = Batch.from_data_list(re_graph_list)
        return graph_batch      
        

class nconv(torch.nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A = A.to(x.device)

        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(torch.nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
    
class gcn(torch.nn.Module):
    def __init__(self,c_in, c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = 3*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x , support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            # print(x1.shape)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        # print(h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    

class GraphWaveNet(torch.nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, supports, batch_size, dropout=0.3, adaptadj=True, in_dim=1,out_dim=100,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):
        """
            kindly note that although there is a 'supports' parameter, we will not use the prior graph if there is a learned dependency graph.
            Details can be found in the feed forward function.
        """
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adaptadj = adaptadj

        self.filter_convs = torch.nn.ModuleList()
        self.gate_convs = torch.nn.ModuleList()
        self.residual_convs = torch.nn.ModuleList()
        self.skip_convs = torch.nn.ModuleList()
        # self.bn = nn.ModuleList()
        self.gconv = torch.nn.ModuleList()
        self.fc_his = torch.nn.Sequential(torch.nn.Linear(96, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU())
        self.start_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        # print("supports:{}".format(supports))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 2
        if supports is not None:
            self.supports_len += len(supports)

        
        self.nodevec1 = nn.Parameter(torch.randn(batch_size, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, batch_size), requires_grad=True)

        self.supports_len +=1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(torch.nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(torch.nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(torch.nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(torch.nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                # self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = torch.nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = torch.nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field
        # print(self.nodevec1.shape)

    def _calculate_random_walk_matrix(self, adj_mx):
        if len(adj_mx.shape) == 3:
            B, N, N = adj_mx.shape

            adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
            d = torch.sum(adj_mx, 2)
            d_inv = 1. / d
            d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
            d_mat_inv = torch.diag_embed(d_inv)
            random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        else:
            N, N = adj_mx.shape
            adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).to(adj_mx.device)
            d = torch.sum(adj_mx, 1)
            d_inv = 1. / d
            d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
            d_mat_inv = torch.diag_embed(d_inv)
            random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, input, hidden_states=None, sampled_adj=None, adp=None, return_st = False):
        input = input.transpose(1, 3)
        input = torch.nn.functional.pad(input,(1,0,0,0))

        input = input[:, :2, :, :]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = torch.nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        new_supports = copy.deepcopy(self.supports)
        if sampled_adj is not None:
            new_supports += [self._calculate_random_walk_matrix(sampled_adj)]
            new_supports += [self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]

        if self.adaptadj:
            # print(self.vec1.shape)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if new_supports is None:
                new_supports=[adp]
            else:
                new_supports += [adp]
        temporal_only = []
        spatio_temporal = []
        
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # gate = torch.unsqueeze(gate, dim=0)
            gate = torch.sigmoid(gate)
            x = filter * gate

            if i != self.blocks * self.layers:
                temporal_only.append(x)
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            if i != self.blocks * self.layers:
                spatio_temporal.append(x)
            x = x + residual[:, :, :, -x.size(3):]

        if hidden_states is not None:
            hidden_states = self.fc_his(hidden_states)        # B, N, D
            hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
            skip = skip + hidden_states
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        if not return_st:
            return x
        else:
            return x, temporal_only, spatio_temporal

    
class FrontAndHead(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()
        print(hid_dim)
        self.subgraphgcn = SubPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)
        self.gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type='TransformerConv')
        self.adj_s = []
        self.gwn = GraphWaveNet(self.adj_s, batch_size, dropout=0.3)
        self.pre_flow_node = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.LeakyReLU())
        self.pre_flow_od = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.LeakyReLU())
        self.pre_flow_subgraph = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.LeakyReLU())
        
        self.W = torch.nn.Parameter(torch.randn(hid_dim, 100))
        self.B = torch.nn.Parameter(torch.randn(100))
                    
        
        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch,
                flow_matrix,
                node_feature, 
                node_seq, 
                stgnn, 
                task,
                device):
        
        row_indices = graph_batch.flag 
        col_indices = graph_batch.times
        if task == 'od':
            selected_rows = torch.zeros((len(row_indices), flow_matrix.shape[0]))
            for index, element_index in enumerate(row_indices):
                selected_rows[index] = flow_matrix[:, element_index[0], element_index[1]]
            times_series = torch.zeros((col_indices.shape[0], 12)).to(device)
            actual_value = torch.zeros((col_indices.shape[0], 1)).to(device)
            for index in range(col_indices.shape[0]):
                times_series[index] = selected_rows[index][col_indices[index]:col_indices[index]+12]
                actual_value[index] = selected_rows[index][col_indices[index]+12:col_indices[index]+13]
            times_series = torch.unsqueeze(times_series, dim=-1).permute(1, 0, 2)
            times_series = torch.unsqueeze(times_series, dim=0)
        elif task == 'subgraph' or task == 'node':
            selected_rows = torch.index_select(flow_matrix, 1, row_indices).permute(1, 0)
            times_series = torch.zeros((col_indices.shape[0], 12)).to(device)
            actual_value = torch.zeros((col_indices.shape[0], 1)).to(device)
            for index in range(col_indices.shape[0]):
                times_series[index] = selected_rows[index][col_indices[index]:col_indices[index]+12]
                actual_value[index] = selected_rows[index][col_indices[index]+12:col_indices[index]+13]
            times_series = torch.unsqueeze(times_series, dim=-1).permute(1, 0, 2)
            times_series = torch.unsqueeze(times_series, dim=0)
        prompted_graph = self.subgraphgcn(graph_batch) 
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        
        embedding_time = self.gwn(times_series)
        embedding_time = torch.squeeze(embedding_time, dim=0)
        final_embedding =  embedding_time * torch.sigmoid(torch.matmul(graph_emb, self.W) + self.B)
        
        if task == 'node':
            predict_value = self.pre_flow_node(final_embedding)
        elif task == 'od':
            predict_value = self.pre_flow_od(final_embedding)
        elif task == 'subgraph':
            predict_value = self.pre_flow_subgraph(final_embedding)
        pre = self.answering(graph_emb)
       

        return pre, prompted_graph.y, predict_value, actual_value



