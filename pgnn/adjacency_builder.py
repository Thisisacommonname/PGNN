import numpy as np


def load_gmt(gmt_file):
    pathways = []
    genes = []

    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            pw = parts[0]
            gene_list = parts[2:]  # pathway genes
            pathways.append(pw)
            genes.append(gene_list)

    return pathways, genes


def build_adjacency(gmt_file, gene_order=None):
   #build pathways * genes
    pathways, gene_sets = load_gmt(gmt_file)

    #flatten from GMT
    all_genes = sorted(list(set(sum(gene_sets, []))))

    # use gmt genes if none 
    if gene_order is None:
        gene_order = all_genes

    gene_order = list(gene_order)

    #map genes with indices
    gene_idx = {g: i for i, g in enumerate(gene_order)}

    # create adjacency matrix
    A = np.zeros((len(pathways), len(gene_order)), dtype=np.float32)

    for i, gset in enumerate(gene_sets):
        for g in gset:
            if g in gene_idx:
                A[i, gene_idx[g]] = 1.0

    return pathways, gene_order, A