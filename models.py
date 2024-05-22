import torch
import torch.nn as nn
import torch.nn.functional as F

import args
from layers import Linear, MLP_encoder, MLP_decoder
from itertools import chain
from utils import feature_propagation, compute_similarity, get_knn_graph
from layers import GraphConvolution


#Attribute Module
class ATTModule(nn.Module):
    def __init__(self, in_channels, hid_channels, dropout=0.3):
        super(attModule, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels, hid_channels),
                                                 )
        self.encoder = MLP_encoder(nfeat=in_channels,
                           nhid=hid_channels,
                           dropout=dropout)

        self.decoder = MLP_decoder(nhid=hid_channels,
                           nfeat=in_channels,
                           dropout=dropout)
        self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        # self.lin.extend(list(self.linear_cls_out.parameters()))

        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, args, x, adj):
        scale_l = []
        for i in range(args.l):
            x_i = feature_propagation(adj, x, i, args.alpha)
            x_i = self.encoder(x_i)
            scale_l.append(x_i)

        h_filters = torch.stack(scale_l, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.filter_num):
            res += h_filters[:, i, :] * score[:, i]

        res = self.decoder(res)

        return res


    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
               (test_attn_anomaly, test_attn_normal)



#Structure Module
class STRModule(nn.Module):
    def __init__(self, nfeat, nhid, dropout, use_bn=False):
        super(strModule, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                   nhid=nhid,
                                   dropout=dropout)

        self.encoder_2 = MLP_encoder(nfeat=nfeat,
                                   nhid=nhid,
                                   dropout=dropout)

        self.decoder = MLP_decoder(nhid=nhid,
                                   nfeat=nfeat,
                                   dropout=dropout)

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, args, x, adj):
        knn = get_knn_graph(x, args.k)

        h1 = feature_propagation(adj, x, args.l, args.alpha)
        h1 = self.encoder(h1)

        h2 = feature_propagation(knn, x, args.l, args.alpha)
        h2 = self.encoder(h2)

        h = torch.concat(self.encoder_2(h1), self.encoder_2(h2))

        res = self.decoder(h)

        return h1, h2, res @ res.T


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


class attModule(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(attModule, self).__init__()

        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)

    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # return reconstructed matrices
        return x_hat


class strModule(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(strModule, self).__init__()

        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        # struct_reconstructed = compute_similarity(x_hat)
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        return x_hat, struct_reconstructed, x, x

