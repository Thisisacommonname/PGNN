# pgnn/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PathwayGraphLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.self_fc = nn.Linear(dim, dim)
        self.neigh_fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        deg = A.sum(dim=1, keepdim=True) + 1e-8
        A_norm = A / deg
        neigh = torch.matmul(A_norm.unsqueeze(0), x)

        h = self.self_fc(x) + self.neigh_fc(neigh)
        h = F.relu(h)
        h = self.dropout(h)
        return self.ln(x + h)


class PathwayGraphNetwork(nn.Module):
    def __init__(self, dim, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PathwayGraphLayer(dim, dropout)
            for _ in range(layers)
        ])

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x


class GeneToPathwayAttention(nn.Module):
    def __init__(self, gene_dim, pathway_dim, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(gene_dim, pathway_dim)
        self.value = nn.Linear(gene_dim, pathway_dim)
        self.dropout = nn.Dropout(dropout)
        self.pathway_query = None

    def build_queries(self, n_pathways, pathway_dim, device):
        if self.pathway_query is None or self.pathway_query.shape[0] != n_pathways:
            q = torch.randn(n_pathways, pathway_dim, device=device) * 0.02
            self.pathway_query = nn.Parameter(q)

    def forward(self, gene_emb, A):
        B, G, _ = gene_emb.shape
        P, G2 = A.shape
        assert G == G2

        Dp = self.value.out_features
        self.build_queries(P, Dp, gene_emb.device)

        K = self.key(gene_emb)
        V = self.value(gene_emb)
        Q = self.pathway_query.unsqueeze(0).expand(B, P, Dp)

        logits = torch.einsum("bpd,bgd->bpg", Q, K) / (Dp ** 0.5)
        mask = (A.unsqueeze(0) == 0)
        logits = logits.masked_fill(mask, -1e9)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        pw_emb = torch.einsum("bpg,bgd->bpd", attn, V)
        return pw_emb, attn


class PathwayToCellAttention(nn.Module):
    def __init__(self, pathway_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(pathway_dim) * 0.02)

    def forward(self, pw):
        logits = torch.einsum("d,bpd->bp", self.query, pw) / (pw.shape[-1] ** 0.5)
        scores = F.softmax(logits, dim=1)
        cell = torch.einsum("bp,bpd->bd", scores, pw)
        return cell, scores


class PGNN_v5(nn.Module):
    def __init__(self, A, P_adj, n_genes, n_pathways, n_classes, config):
        super().__init__()

        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float32)
        if isinstance(P_adj, np.ndarray):
            P_adj = torch.tensor(P_adj, dtype=torch.float32)

        self.register_buffer("A", A)
        self.register_buffer("P_adj", P_adj)

        self.gene_embed = nn.Embedding(n_genes, config.GENE_DIM)
        self.expr_proj = nn.Linear(1, config.GENE_DIM)
        self.gene_ln = nn.LayerNorm(config.GENE_DIM)
        self.gene_dropout = nn.Dropout(config.DROPOUT)

        self.gene_mlp = nn.Sequential(
            nn.Linear(config.GENE_DIM, config.GENE_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.GENE_DIM, config.GENE_DIM)
        )
        self.gene_mlp_ln = nn.LayerNorm(config.GENE_DIM)

        self.gene_to_pw = GeneToPathwayAttention(
            gene_dim=config.GENE_DIM,
            pathway_dim=config.PATHWAY_DIM,
            dropout=config.DROPOUT
        )

        self.pw_gnn = PathwayGraphNetwork(
            dim=config.PATHWAY_DIM,
            layers=config.PW_GNN_LAYERS,
            dropout=config.DROPOUT
        )

        self.pw_to_cell = PathwayToCellAttention(config.PATHWAY_DIM)

        self.cls_head = nn.Sequential(
            nn.Linear(config.PATHWAY_DIM, config.PATHWAY_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.PATHWAY_DIM, n_classes)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(config.PATHWAY_DIM, config.PATHWAY_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.PATHWAY_DIM, 1)
        )

    def encode_genes(self, x):
        B, G = x.shape
        ids = torch.arange(G, device=x.device).unsqueeze(0).expand(B, G)

        gene_id = self.gene_embed(ids)
        expr = self.expr_proj(x.unsqueeze(-1))

        g = gene_id + expr
        h = self.gene_mlp(g)
        g = self.gene_mlp_ln(g + h)
        g = self.gene_ln(g)
        g = self.gene_dropout(g)
        return g

    def encode_pathways(self, x):
        g = self.encode_genes(x)
        pw, gene_attn = self.gene_to_pw(g, self.A)
        pw = self.pw_gnn(pw, self.P_adj)
        return pw, gene_attn

    def encode_cell(self, x):
        pw, gene_attn = self.encode_pathways(x)
        cell, pathway_attn = self.pw_to_cell(pw)
        return cell, pw, gene_attn, pathway_attn

    def forward(self, x):
        cell, pw, gene_attn, pathway_attn = self.encode_cell(x)
        pred_cls = self.cls_head(cell)
        pred_reg = self.reg_head(cell).squeeze(-1)
        return pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn