import sys
import networkx as nx
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
import pickle as pkl
from loss import reconstruction_loss, L2_loss, triplet_loss


def to_tensor(csc_matrix_data):
    coo_matrix_data = csc_matrix_data.tocoo()
    dense_tensor = torch.tensor(coo_matrix_data.toarray())
    del coo_matrix_data
    return dense_tensor

def compute_similarity(attr, adj):
    f = False
    if attr.is_cuda:
        attr = attr.cpu().detach().numpy()
        f = True
    distances = pairwise_distances(attr, metric='euclidean')
    threshold = 0.15
    reconstructed_adj = torch.tensor((distances <= threshold).astype(int))
    if f:
        reconstructed_adj = reconstructed_adj.cuda()
    return reconstructed_adj


def compute_dot(attr, adj):
    f = False
    if attr.is_cuda:
        attr = attr.cpu().detach().numpy()
        f = True
    reconstructed_adj = np.dot(attr, attr.T)
    reconstructed_adj = torch.tensor(reconstructed_adj)
    if f:
        reconstructed_adj = reconstructed_adj.cuda()
    return reconstructed_adj

def auc_roc(y_true, y_pred_probs):
    auc_roc = roc_auc_score(y_true, y_pred_probs)
    return auc_roc

def evaluate_att(args, reconstructed_attr, true_attr, label):
    rec = L2_loss(reconstructed_attr, true_attr)
    ar = auc_roc(label, rec)
    return ar

def evaluate_str(args, reconstructed_adj, true_adj, label):
    rec = L2_loss(reconstructed_adj, true_adj)
    ar = auc_roc(label, rec)
    return ar

def evaluate(args, reconstructed_attr, attr, reconstructed_adj, adj, label):
    # neighbor_avg = torch.matmul(adj, attr) / adj.sum(dim=1, keepdim=True)
    # reconstructed_attr = neighbor_avg
    # reconstructed_adj = compute_dot(attr)

    rec, attr_rec, str_rec, loss = reconstruction_loss(reconstructed_attr, attr, reconstructed_adj, adj, 1, 1, args.lambd)
    ar = auc_roc(label, rec.cpu().detach().numpy())
    return ar

def feature_propagation(adj, features, K, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for i in range(1, K + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()

def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_knn_graph(x, num_neighbor, batch_size=0, knn_metric='cosine', connected_fast=True):
    if not batch_size:
        adj_knn = kneighbors_graph(x, num_neighbor, metric=knn_metric)
    else:
        if connected_fast:
            print('compute connected fast knn')
            num_neighbor1 = int(num_neighbor / 2)
            batches1 = get_random_batch(x.shape[0], batch_size)
            row1, col1 = global_knn(x, num_neighbor1, batches1, knn_metric)
            num_neighbor2 = num_neighbor - num_neighbor1
            batches2 = get_random_batch(x.shape[0], batch_size)
            row2, col2 = global_knn(x, num_neighbor2, batches2, knn_metric)
            row, col = np.concatenate((row1, row2)), np.concatenate((col1, col2))
        else:
            print('compute fast knn')
            batches = get_random_batch(x.shape[0], batch_size)
            row, col = global_knn(x, num_neighbor, batches, knn_metric)
        adj_knn = coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))

    return adj_knn.tolil()

def get_random_batch(n, batch_size):
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    batches = []
    i = 0
    while i + batch_size * 2 < n:
        batches.append(idxs[i:i + batch_size])
        i += batch_size
    batches.append(idxs[i:])
    return batches

def global_knn(x, num_neighbor, batches, knn_metric):
    row = None
    for batch in batches:
        knn_current = kneighbors_graph(x[batch], num_neighbor, metric=knn_metric).tocoo()
        row_current = batch[knn_current.row]
        col_current = batch[knn_current.col]
        if row is None:
            row = row_current
            col = col_current
        else:
            row = np.concatenate((row, row_current))
            col = np.concatenate((col, col_current))
    return row, col


def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask

def apply_feature_mask(features, mask):
    features[mask] = float('nan')

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citetion(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("new_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("new_data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def edge_delete(prob_del, adj, enforce_connected=False):
    rnd = np.random.RandomState(1234)
    adj= adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)
    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()
    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
    del_adj= sp.csr_matrix(del_adj)

    return del_adj

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def mask_data(att, adj, device, attrate, strrate):

    features = sp.csr_matrix(att, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    mask = feature_mask(features, attrate)
    apply_feature_mask(features, mask)

    sedges = np.array(adj, dtype=np.int32).reshape(adj.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = edge_delete(strrate, sadj)

    # A
    tadj = torch.FloatTensor(sadj.todense()).to(device)
    # stu_input
    mask_adj = normalize_sparse(sadj + sp.eye(sadj.shape[0]))
    nsadj = torch.FloatTensor(np.array(sadj.todense())).to(device)

    return att, adj
