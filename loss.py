import torch


def attLoss(recon_att, att, recon_adj, adj, lambd):
    rec, attr_rec, str_rec, loss = reconstruction_loss(recon_att, att, recon_adj, adj, 1, 1, lambd)
    return loss

def strLoss(recon_att, att, recon_adj, adj, lambd, H_1, H_1_p, gamma, f):
    rec, attr_rec, str_rec, rec_loss = reconstruction_loss(recon_att, att, recon_adj, adj, 1, 1, lambd)
    cal_loss = L2_loss(H_1, H_1_p, f)
    loss = rec_loss + gamma * cal_loss
    return loss

def attLossKD(eta_1, eta_2):
    loss = attLoss()
    KD_loss = triplet_loss()
    loss = eta_1 * loss + eta_2 * KD_loss
    return loss

def strLossKD(recon_adj, recon_att, H_1, H_1_p, eta_1, eta_2):
    loss = strLoss(recon_adj, recon_att, H_1, H_1_p)
    KD_loss = triplet_loss()
    loss = eta_1 * loss + eta_2 * KD_loss
    return loss

def reconstruction_loss(preds_attribute, labels_attribute, preds_structure, labels_structure, eta, theta, alpha):

    # attribute reconstruction loss
    B_attr = labels_attribute * (eta - 1) + 1
    diff_attribute = torch.square(torch.subtract(preds_attribute, labels_attribute) * B_attr)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, dim=1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    B_struc = labels_structure * (theta - 1) + 1
    diff_structure = torch.square(torch.subtract(preds_structure, labels_structure) * B_struc)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    reconstruction_errors = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    cost = alpha * attribute_cost + (1 - alpha) * structure_cost
    return reconstruction_errors, attribute_reconstruction_errors, structure_reconstruction_errors, cost

def al():
    pass

def sl():
    pass

def L2_loss(X, X_hat, f):
    if f:
        # Attribute reconstruction loss
        diff = torch.pow(X - X_hat, 2)
        errors = torch.sqrt(torch.sum(diff, 1))
    else:
        errors = 0

    return errors

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    loss = torch.mean(losses)
    return loss
