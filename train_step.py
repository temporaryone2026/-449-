# train_step.py
import torch
from losses import pairwise_ranking_loss, kg_contrastive_loss, soft_orthogonal_regularizer
from retriever import StudentRetriever
from gnp import GraphNeuralPrompting

def train_step(student: StudentRetriever, teacher_scores_fn, gnp: GraphNeuralPrompting,
               optimizer, batch, config):
    """
    Single training step for AKD-RAG asymmetric distillation.
    batch should contain:
      q_feats: [B, D]
      d_feats: [B, N, D]
      entity_node_feats_list: length B, each [num_nodes, node_feat_dim]
      entity_adj_list: length B, each [num_nodes, num_nodes]
      student_entity_embs: [B, H] (extracted from student by entity mentions mapping)
    config:
      K (topk), margin, lambda_KG, lambda_ortho, tau
    teacher_scores_fn: function(q_feats,d_feats) -> [B,N] (teacher s_T(q,d))
    """
    student.train()
    optimizer.zero_grad()

    q_feats = batch["q_feats"].to(next(student.parameters()).device)
    d_feats = batch["d_feats"].to(next(student.parameters()).device)

    # 1) teacher top-K behavior (teacher_scores_fn may run teacher model or load precomputed)
    with torch.no_grad():
        t_scores = teacher_scores_fn(q_feats, d_feats)     # [B, N]
        topk_idx = torch.topk(t_scores, config["K"], dim=1).indices  # [B, K]

    # 2) GNP produce prompts per query (batch)
    prompts = []
    for i in range(q_feats.size(0)):
        node_feats = batch["entity_node_feats_list"][i].to(q_feats.device)
        adj = batch["entity_adj_list"][i].to(q_feats.device)
        prompt = gnp(node_feats, adj)   # [prompt_dim]
        prompts.append(prompt)
    prompts = torch.stack(prompts, dim=0)  # [B, prompt_dim]

    # 3) student forward (inject prompt into query)
    q_emb = student.encode_query(q_feats, prompt=prompts)   # [B,H]
    d_emb = student.encode_docs(d_feats)                    # [B,N,H]
    s_scores = student.score(q_emb, d_emb)                  # [B,N]

    # 4) ranking loss (asymmetric distillation applied only to retriever)
    L_rank = pairwise_ranking_loss(s_scores, topk_idx, margin=config.get("margin", 0.1))

    # 5) KG contrastive loss: align student entity reps with GNP prompts
    # student_entity_embs should be prepared (e.g., aggregate token embeddings for linked entities)
    student_entity_embs = batch["student_entity_embs"].to(q_feats.device)  # [B,H]
    kg_prompts = prompts  # [B,H]
    L_KG = kg_contrastive_loss(student_entity_embs, kg_prompts, temperature=config.get("tau", 0.07))

    # 6) orthogonal regularizer
    L_ortho = soft_orthogonal_regularizer(q_emb, kg_prompts)

    # 7) joint loss
    loss = L_rank + config.get("lambda_KG", 1.0) * L_KG + config.get("lambda_ortho", 0.1) * L_ortho
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_rank": L_rank.item(),
        "L_KG": L_KG.item(),
        "L_ortho": L_ortho.item()
    }