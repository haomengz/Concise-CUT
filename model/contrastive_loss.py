import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.manual_seed(12345)
cross_entropy_loss = torch.nn.CrossEntropyLoss()

def patch_nce_loss(feat_q, feat_k, tau=0.07):
    # Patch_nce_loss
    # ref: https://arxiv.org/pdf/2007.15651.pdf
    B, C, S = feat_q.shape
    l_pos = (feat_k * feat_q).sum(dim=1)[:, :, None]
    l_neg = torch.bmm(feat_q.transpose(1, 2), feat_k)

    # Mask out the diagonal elements as they represent the similarity between themselves.
    # Here we need to create a ByteTensor to make the masked_fill_() function work.
    # The pesudo code provided does not specify dtype to create the mask, which does not work during the testing.
    identity_matrix = torch.eye(S, dtype=torch.bool)[None, :, :]
    l_neg = l_neg.masked_fill_(identity_matrix, -float('inf'))
    logits = (torch.cat((l_pos, l_neg), dim=2) / tau).flatten(0, 1)
    return cross_entropy_loss(predictions, torch.zeros(B * S, dtype=torch.long))


def Springlike_loss(feat_q, feat_k, m=1):
    # Springlike loss
    # ref: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    B, C, S = feat_q.shape
    current_loss = 0
    for s in range(S):
        current_feat_q = feat_q[:, :, s][:, :, None]
        l2_norm = (current_feat_q - feat_k).norm(dim=-2, keepdim=True)
        l2_norm[:, :, :s] = F.relu(m - l2_norm[:, :, :s])
        l2_norm[:, :, s+1:] = F.relu(m - l2_norm[:, :, s+1:])
        current_loss += l2_norm.sum()
    return current_loss / 2


def Triplet_loss(feat_q, feat_k, alpha=1):
    # Triplet loss
    # ref: https://arxiv.org/pdf/1503.03832.pdf
    B, C, S = feat_q.shape
    current_loss = 0
    for s in range(S):
        current_feat_q = feat_q[:, :, s][:, :, None]
        l2_norm = (current_feat_q - feat_k).norm(dim=-2, keepdim=True)
        if s == 0:
            current_loss += F. relu(- l2_norm[:, :, s+1] + l2_norm[:, :, s] + alpha).sum()
        else:
            current_loss += F. relu(-l2_norm[:, :, s-1] + l2_norm[:, :, s] + alpha).sum()
    return current_loss


def patch_nce_loss_test():
    feat_q = torch.randn(10, 3, 100)
    feat_k = torch.randn(10, 3, 100)
    print(patch_nce_loss(feat_q, feat_k))


def Springlike_loss_test():
    feat_q = torch.randn(10, 3, 100)
    feat_k = torch.randn(10, 3, 100)
    print(Springlike_loss(feat_q, feat_k))


def Triplet_loss_test():
    feat_q = torch.randn(10, 3, 100)
    feat_k = torch.randn(10, 3, 100)
    print(Triplet_loss(feat_q, feat_k))
