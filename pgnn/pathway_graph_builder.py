# pgnn/pathway_graph_builder.py
import numpy as np


def build_pathway_graph(A, min_jaccard=0.08, self_loop=True):
    #pathway gene adjacency, sahpe in A,P
    A = (A > 0).astype(np.float32)
    P = A.shape[0]

    P_adj = np.zeros((P, P), dtype=np.float32)

    for i in range(P):
        gi = A[i].astype(bool)
        for j in range(i, P):
            gj = A[j].astype(bool)

            inter = np.logical_and(gi, gj).sum()
            union = np.logical_or(gi, gj).sum()

            score = 0.0 if union == 0 else inter / union

            if score >= min_jaccard:
                P_adj[i, j] = 1.0
                P_adj[j, i] = 1.0

    if self_loop:
        np.fill_diagonal(P_adj, 1.0)

    return P_adj