# gnp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N] (float)
        h = torch.matmul(adj, x)  # aggregate neighbor features
        return F.relu(self.linear(h))

class GraphNeuralPrompting(nn.Module):
    """
    GNP: encode a KG subgraph (entities + relations as adj matrix) into a soft prompt vector.
    - node_feats: [num_nodes, node_feat_dim]
    - adj: [num_nodes, num_nodes] (symmetric)
    Produces:
    - prompt: [prompt_dim] (or [batch, prompt_dim])
    """
    def __init__(self, node_feat_dim=128, hidden_dim=256, prompt_dim=384, pool="avg"):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(node_feat_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, prompt_dim)
        self.pool = pool

    def forward(self, node_feats, adj):
        """
        node_feats: [num_nodes, feat_dim]
        adj: [num_nodes, num_nodes]
        returns: prompt [prompt_dim]
        """
        h = self.gcn1(node_feats, adj)       # [N, hidden]
        h = self.gcn2(h, adj)                # [N, hidden]
        if self.pool == "avg":
            pooled = h.mean(dim=0)           # [hidden]
        elif self.pool == "max":
            pooled, _ = h.max(dim=0)
        else:
            pooled = h.mean(dim=0)
        prompt = self.proj(pooled)           # [prompt_dim]
        return prompt