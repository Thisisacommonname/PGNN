# pgnn/dataloader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from config import Config


class PGNN_Dataset(Dataset):
    def __init__(
        self,
        expr_file,
        meta_file,
        gene_order,
        adj_matrix=None,
        pathways=None,
        top_var_genes=None,
    ):
        expr = pd.read_csv(expr_file, index_col=0)
        expr = expr.T   #transpose genes * samples

        meta = pd.read_csv(meta_file)

        meta = meta.set_index("sample")

        # align samples
        common_samples = expr.index.intersection(meta.index)
    
        expr = expr.loc[common_samples]
        meta = meta.loc[common_samples]
        #exclusion
        if len(Config.EXCLUDE_GROUPS) > 0:
            keep_mask = ~meta["group"].isin(Config.EXCLUDE_GROUPS)
            expr = expr.loc[keep_mask]
            meta = meta.loc[keep_mask]

        self.sample_names = meta.index.tolist()

        # regression target
        if "fidelity" in meta.columns:
            y_reg = meta["fidelity"].astype(float).values
        else:
            # assign each samples a weight for fidelity caculation
            y_reg = []
            for g in meta["group"].astype(str):
                if g in Config.PRIMARY_GROUPS:
                    y_reg.append(1.0)
                elif "hepatoblast" in g.lower():
                    y_reg.append(0.0)
                elif "aav_rescue" in g.lower():
                    y_reg.append(0.65)
                elif "3d_" in g.lower():
                    y_reg.append(0.55)
                elif "deletion" in g.lower():
                    y_reg.append(0.25)
                else:
                    y_reg.append(0.45)
            y_reg = np.asarray(y_reg, dtype=np.float32)

       #primary vs. non-primary
        y_cls = np.where(meta["group"].isin(Config.PRIMARY_GROUPS), 1, 0)

        meta["y_reg"] = y_reg
        meta["y_cls"] = y_cls
        meta["label_name"] = np.where(meta["y_cls"] == 1, "primary", "non_primary")
        self.meta = meta.copy()


        # variance filter
        if top_var_genes is not None and top_var_genes < expr.shape[1]:
            gene_var = expr.var(axis=0)
            keep_var = gene_var.sort_values(ascending=False).head(top_var_genes).index.tolist()
            expr = expr[keep_var]
    

        #align to pathway genes
        expr_genes = expr.columns.tolist()
        keep_genes = [g for g in gene_order if g in expr_genes]

        gene_index = {g: i for i, g in enumerate(expr_genes)}
        X = np.zeros((expr.shape[0], len(keep_genes)), dtype=np.float32)

        for j, g in enumerate(keep_genes):
            X[:, j] = expr.iloc[:, gene_index[g]].values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.gene_order = keep_genes

        self.y_reg = torch.tensor(meta["y_reg"].values, dtype=torch.float32)
        self.y_cls = torch.tensor(meta["y_cls"].values, dtype=torch.long)

        self.classes = ["non_primary", "primary"]
        self.n_classes = 2

        self.adj_matrix = adj_matrix
        self.pathways = pathways


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]