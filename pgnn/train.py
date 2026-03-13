import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from model import PGNN_v5
from dataloader import PGNN_Dataset
from adjacency_builder import build_adjacency
from pathway_graph_builder import build_pathway_graph


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():

    set_seed(Config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pathways, gene_order, A_full = build_adjacency(Config.GMT_FILE)

    expr_genes = pd.read_csv(Config.EXPR_FILE, index_col=0).index.tolist()
    common_genes = [g for g in gene_order if g in expr_genes]

    gene_index = {g: i for i, g in enumerate(gene_order)}
    keep_idx = [gene_index[g] for g in common_genes]

    A_full = A_full[:, keep_idx]
    gene_order = common_genes
    A_full = A_full.astype(np.float32)


    dataset = PGNN_Dataset(
        Config.EXPR_FILE,
        Config.META_FILE,
        gene_order,
        adj_matrix=A_full,
        pathways=pathways,
        top_var_genes=Config.TOP_VAR_GENES,
    )

    filtered_gene_order = dataset.gene_order

    gene_index2 = {g: i for i, g in enumerate(gene_order)}
    keep_idx2 = [gene_index2[g] for g in filtered_gene_order]

    A = A_full[:, keep_idx2]

    P_adj = build_pathway_graph(
        A,
        min_jaccard=Config.PW_GRAPH_JACCARD,
        self_loop=True
    ).astype(np.float32)

    y = dataset.y_cls.numpy()
    idx = np.arange(len(dataset))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=Config.RANDOM_SEED
    )

    train_idx, val_idx = next(splitter.split(idx, y))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )


    model = PGNN_v5(
        A=A,
        P_adj=P_adj,
        n_genes=len(filtered_gene_order),
        n_pathways=len(pathways),
        n_classes=dataset.n_classes,
        config=Config,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=6
    )

    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    best_val = float("inf")
    wait = 0


    for epoch in range(Config.EPOCHS):

        model.train()
        train_loss = 0.0

        for x, y_reg, y_cls in train_loader:

            x = x.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn = model(x)

            loss = (
                Config.LAMBDA_CLS * loss_cls(pred_cls, y_cls)
                + Config.LAMBDA_REG * loss_reg(pred_reg, y_reg)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for x, y_reg, y_cls in val_loader:

                x = x.to(device)
                y_reg = y_reg.to(device)
                y_cls = y_cls.to(device)

                pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn = model(x)

                loss = (
                    Config.LAMBDA_CLS * loss_cls(pred_cls, y_cls)
                    + Config.LAMBDA_REG * loss_reg(pred_reg, y_reg)
                )

                val_loss += loss.item()

                preds = pred_cls.argmax(dim=1)

                correct += (preds == y_cls).sum().item()

                total += y_cls.size(0)

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)

        val_acc = correct / total if total > 0 else 0.0

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} "
            f"- TrainLoss {train_loss:.4f} "
            f"- ValLoss {val_loss:.4f} "
            f"- ValAcc {val_acc:.3f}"
        )

        if val_loss < best_val:

            best_val = val_loss
            wait = 0

            torch.save(
                model.state_dict(),
                Config.MODEL_PATH
            )

        else:

            wait += 1

            if wait >= Config.PATIENCE:

                break


    #cell embeddings

    print("\nSaving cell embeddings...")

    model.load_state_dict(torch.load(Config.MODEL_PATH))

    model.eval()

    all_embeddings = []
    all_samples = []

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    with torch.no_grad():

        for i, (x, y_reg, y_cls) in enumerate(loader):

            x = x.to(device)

            pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn = model(x)

            all_embeddings.append(cell.cpu().numpy())

            start = i * Config.BATCH_SIZE
            end = start + x.shape[0]

            all_samples.extend(
                dataset.meta.index[start:end].tolist()
            )

    embeddings = np.vstack(all_embeddings)

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(
        embeddings,
        index=all_samples
    )

    df.to_csv("results/cell_embeddings.csv")


if __name__ == "__main__":

    train()