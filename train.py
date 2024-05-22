from __future__ import division
from __future__ import print_function
import scipy.io as sio
import torch
import torch.optim as optim
from args import args
from models import attModule, strModule
from utils import evaluate, feature_propagation, process_adj, get_knn_graph, \
    mask_data, evaluate_att, evaluate_str, compute_similarity, compute_dot
from loss import attLoss, strLoss, attLossKD, strLossKD

# Model and optimizer
class TripleAD:
    def __init__(self, args):
        self.args = args
        self.best_att_auc, self.best_str_auc = 0, 0
        self.load_data()
        self.att_state, self.str_state = None, None

        # Model Initialization
        self.att_model = attModule(feat_size = self.attr.size(1), hidden_size = 128, dropout = 0.3)
        self.att_model.to(args.device)

        self.str_model = strModule(feat_size = self.attr.size(1), hidden_size = 128, dropout = 0.3)
        self.str_model.to(args.device)

        self.f = None
        # Setup Training Optimizer
        self.optimizerAtt = optim.Adam(self.att_model.parameters(), lr=self.args.learning_rate,
                                              weight_decay=self.args.weight_decay_att)
        self.optimizerStr = optim.Adam(self.str_model.parameters(), lr=self.args.learning_rate,
                                              weight_decay=self.args.weight_decay_str)


    def load_data(self):
        dataset = self.args.dataset
        graph = sio.loadmat('./dataset/{}.mat'.format(dataset))
        self.attr = torch.tensor(graph['Attributes'].toarray()).float()
        self.adj = torch.tensor(graph['Network'].toarray()).float()
        self.label = graph['Label']

        self.attr, self.adj = mask_data(self.attr, self.adj, self.args.device, self.args.attMask, self.args.strMask)

        self.attr, self.adj = self.to_device(self.attr, self.adj)
        print("dataset:{}".format(dataset))


    def to_device(self, attr, adj):
        attr = attr.to(self.args.device)
        adj = adj.to(self.args.device)
        return attr, adj


    def pre_strdata(self):
        # Compute x_prop
        self.adj = process_adj(self.adj)
        x_prop = feature_propagation(self.adj, self.att, self.args.propagation_iteration_T, self.args.restart_prob_b)

        # Compute x_prop_aug
        adj_knn = get_knn_graph(x_prop, self.args.k, knn_metric=args.knn_metric)
        adj_knn = process_adj(adj_knn)
        x_prop_aug = feature_propagation(adj_knn, self.att, self.args.propagation_iteration_T, self.args.restart_prob_b)

        return x_prop, x_prop_aug

    def pre_train_str(self):
        if self.f:
            self.strmodel.train()
            self.optimizerAtt.zero_grad()
            G, G_enhanced = self.pre_strdata()
            A_hat, H, H_1 = self.str_model(G, G_enhanced)

            loss_train = self.strLoss(A_hat, H, H_1)
            loss_train.backward()
            self.optimizerAtt.step()
            auc_roc = evaluate_str(args, A_hat, self.adj, self.label)

    def pre_attdata(self):
        self.adj = process_adj(self.adj)
        mu_scal_x = []
        for l in range(self.args.multi_scale_L):
            x_prop = feature_propagation(self.adj, self.att, self.args.propagation_iteration_T, self.args.restart_prob_b)
            mu_scal_x.append(x_prop)
        return mu_scal_x

    def pre_train_att(self):
        if self.f:
            self.attmodel.train()
            muscal_x = self.pre_attdata()
            self.optimizerStr.zero_grad()
            X_hat = self.att_model(muscal_x)

            loss_train = self.attLoss(X_hat)
            loss_train.backward()
            self.optimizerStr.step()

            auc_roc = evaluate_att(args, X_hat, self.attr, self.label)


    def train_attModule(self, epoch):

        self.att_model.train()
        self.optimizerAtt.zero_grad()
        # X_hat, A_hat = self.att_model(self.attr, self.adj)
        X_hat = self.att_model(self.attr, self.adj)
        X_att, A_hat, H_1_p, H_1 = self.str_model(self.attr, self.adj)

        loss_train = attLoss(X_hat, self.attr, A_hat, self.adj, 1)
        loss_train.backward()
        self.optimizerAtt.step()

        if (epoch+1) == self.args.att_epoch:
            auc_roc = evaluate(args, X_hat, self.attr, A_hat, self.adj, self.label)
            print("[AttModule Training] epoch: {}----AUC-ROC: {:.2f}".format(epoch+1, auc_roc*100))

    def train_strModule(self, epoch):

        self.str_model.train()
        self.optimizerStr.zero_grad()
        X_hat = self.att_model(self.attr, self.adj)
        X_att, A_hat, H_1_p, H_1 = self.str_model(self.attr, self.adj)
        if self.args.dataset in ['citeseer']:
            A_hat = compute_similarity(self.attr, A_hat)
        elif self.args.dataset in ['flickr']:
            A_hat = compute_dot(self.attr, A_hat)
            neighbor_avg = torch.matmul(self.adj, self.attr) / self.adj.sum(dim=1, keepdim=True)

        loss_train = strLoss(X_hat, self.attr, A_hat, self.adj, 0, H_1_p, H_1, self.args.gamma, self.f)
        loss_train.backward()
        self.optimizerStr.step()

        if (epoch+1) == self.args.str_epoch:
            auc_roc = evaluate(args, X_hat, self.attr, A_hat, self.adj, self.label)
            print("[StrModule Training] epoch: {}----AUC-ROC: {:.2f}".format(epoch+1, auc_roc*100))

    def test(self):
        X_hat = self.att_model(self.attr, self.adj)
        X_att, A_hat, H_1_p, H_1 = self.str_model(self.attr, self.adj)
        if self.args.dataset in ['citeseer']:
            A_hat = compute_similarity(self.attr, A_hat)
        elif self.args.dataset in ['flickr']:
            A_hat = compute_dot(self.attr, A_hat)
        result = evaluate(args, X_hat, self.attr, A_hat, self.adj, self.label)
        return result*100

    def save_checkpoint(self, filename='./.checkpoints/' + args.dataset, ts='att'):
        # print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'att':
            torch.save(self.att_state, filename)
            # print('Successfully saved feature teacher model\n...')
        elif ts == 'str':
            torch.save(self.str_state, filename)
            # print('Successfully saved structure teacher model\n...')


    def load_checkpoint(self, filename='./.checkpoints/' + args.dataset, ts='att'):
        # print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'att' and self.f:
            load_state = torch.load(filename)
            self.att_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherStr.load_state_dict(load_state['optimizer'])
        elif ts == 'str' and self.f:
            load_state = torch.load(filename)
            self.str_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherStr.load_state_dict(load_state['optimizer'])


