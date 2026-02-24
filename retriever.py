# retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentRetriever(nn.Module):
    """
    Minimal student retriever showing how to inject a prompt.
    Replace query_encoder/doc_encoder with BERT/DPR for real experiments.
    """
    def __init__(self, text_feat_dim=768, hidden_dim=384):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.doc_encoder = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def encode_query(self, q_feats, prompt=None):
        # q_feats: [B, D]; prompt: [B, H] or [H]
        q_emb = self.query_encoder(q_feats)  # [B, H]
        if prompt is not None:
            # add soft prompt (element-wise addition)
            if prompt.dim() == 1:
                q_emb = q_emb + prompt.unsqueeze(0)  # broadcast
            else:
                q_emb = q_emb + prompt
        return q_emb  # [B, H]

    def encode_docs(self, d_feats):
        # d_feats: [B, N, D]
        B, N, D = d_feats.size()
        flat = d_feats.view(B*N, D)
        enc = self.doc_encoder(flat).view(B, N, -1)  # [B, N, H]
        return enc

    def score(self, q_emb, d_emb):
        # q_emb: [B, H], d_emb: [B, N, H] -> [B, N]
        scores = torch.einsum("bh,bnh->bn", q_emb, d_emb)
        return scores