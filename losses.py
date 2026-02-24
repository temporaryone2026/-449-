# losses.py
import torch
import torch.nn.functional as F

def pairwise_ranking_loss(student_scores, teacher_topk_idx, margin=0.1):
    """
    Pairwise ranking loss (hinge) approximates Top-K ranking consistency.
    Formula:
      L_ret = (1 / |B|) * sum_q ( 1/(|T_q|*m) * sum_{d+ in T_q} sum_{d- sampled} max(0, margin - s_s(q,d+) + s_s(q,d-)) )
    Inputs:
      student_scores: [B, N] (s_S(q,d))
      teacher_topk_idx: [B, K]
    """
    B, N = student_scores.shape
    loss = 0.0
    count = 0
    for b in range(B):
        pos_idx = teacher_topk_idx[b]            # [K]
        mask = torch.ones(N, dtype=torch.bool, device=student_scores.device)
        mask[pos_idx] = False
        neg_idx = torch.nonzero(mask).squeeze(1)
        if neg_idx.numel() == 0:
            continue
        s_b = student_scores[b]
        s_pos = s_b[pos_idx]                     # [K]
        # sample at most K negatives
        k = min(pos_idx.numel(), neg_idx.numel())
        perm = torch.randperm(neg_idx.numel(), device=student_scores.device)[:k]
        sampled_neg = neg_idx[perm]
        s_neg = s_b[sampled_neg]                 # [k]
        # compute hinge for all pairs
        # margin - s_pos[:,None] + s_neg[None,:]
        delta = margin - s_pos.unsqueeze(1) + s_neg.unsqueeze(0)
        loss += torch.clamp(delta, min=0.0).sum()
        count += delta.numel()
    if count == 0:
        return torch.tensor(0.0, device=student_scores.device)
    return loss / count

def kg_contrastive_loss(student_entity_embs, kg_prompt_embs, temperature=0.07):
    """
    InfoNCE between student entity embeddings e^S and GNP prompt embeddings e^{KG}.
    Formula:
      L_KG = - sum_i log( exp(sim(e^S_i, e^{KG}_i)/tau) / sum_j exp(sim(e^S_i, e^{KG}_j)/tau) )
    Inputs:
      student_entity_embs: [B, H]
      kg_prompt_embs:      [B, H]
    """
    s = F.normalize(student_entity_embs, dim=1)
    t = F.normalize(kg_prompt_embs, dim=1)
    logits = torch.matmul(s, t.t()) / temperature  # [B, B]
    labels = torch.arange(s.size(0), device=s.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def soft_orthogonal_regularizer(repr_ret, repr_kg):
    """
    Soft orthogonal regularizer:
      L_ortho = || repr_ret^T repr_kg ||_F^2 / (H_ret * H_kg)
    repr_ret: [B, H1], repr_kg: [B, H2]
    """
    r1 = repr_ret - repr_ret.mean(dim=0, keepdim=True)
    r2 = repr_kg - repr_kg.mean(dim=0, keepdim=True)
    mat = torch.matmul(r1.t(), r2)  # [H1, H2]
    loss = (mat**2).sum() / (repr_ret.size(1) * repr_kg.size(1))
    return loss