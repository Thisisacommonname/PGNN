# pgnn/interpret_pgnn.py
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from config import Config
from model import PGNN_v5
from dataloader import PGNN_Dataset
from adjacency_builder import build_adjacency
from pathway_graph_builder import build_pathway_graph


OUTDIR = Config.ANALYSIS_DIR
os.makedirs(OUTDIR, exist_ok=True)


def load_model(device):
    pathways, gene_order, A = build_adjacency(Config.GMT_FILE)

    dataset = PGNN_Dataset(
        Config.EXPR_FILE,
        Config.META_FILE,
        gene_order,
        adj_matrix=A,
        pathways=pathways,
        top_var_genes=Config.TOP_VAR_GENES,
    )

    filtered_gene_order = dataset.gene_order
    gene_index = {g: i for i, g in enumerate(gene_order)}
    keep_idx = [gene_index[g] for g in filtered_gene_order]
    A = A[:, keep_idx]

    P_adj = build_pathway_graph(
        A,
        min_jaccard=Config.PW_GRAPH_JACCARD,
        self_loop=True
    ).astype(np.float32)

    model = PGNN_v5(
        A=A,
        P_adj=P_adj,
        n_genes=len(filtered_gene_order),
        n_pathways=len(pathways),
        n_classes=dataset.n_classes,
        config=Config,
    ).to(device)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device), strict=False)
    model.eval()

    return model, dataset, pathways


def extract_embeddings(model, dataset, device):
    X = dataset.X.to(device)
    with torch.no_grad():
        pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn = model(X)
    return pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn


def plot_top_pathways(pathways, pathway_attn):
    arr = pathway_attn.detach().cpu().numpy()
    importance = arr.std(axis=0) * arr.mean(axis=0)

    idx = np.argsort(importance)[::-1][:20]
    df = pd.DataFrame({
        "Pathway": [pathways[i].replace("_", " ") for i in idx],
        "Importance": importance[idx]
    })

    plt.figure(figsize=(13, 7))
    sns.barplot(data=df, x="Importance", y="Pathway", color="#438cb9")
    plt.title("Top Pathways Driving Primary Proximity")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/top_pathways.png", dpi=300)
    plt.close()

    df.to_csv(f"{OUTDIR}/top_pathways.csv", index=False)



def top_genes_global(dataset, gene_attn):
    arr = gene_attn.detach().cpu().numpy()

    if arr.ndim == 3:
        scores = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        scores = arr.mean(axis=0)
    else:
        scores = arr.flatten()

    df = pd.DataFrame({
        "gene": dataset.gene_order,
        "importance": scores
    }).sort_values("importance", ascending=False)

    df.to_csv(f"{OUTDIR}/top_genes_global.csv", index=False)



import textwrap

def clean_pathway_name(name):

    name = name.replace("_", " ")

    remove_words = [
        "GOBP ",
        "KEGG ",
        "REACTOME ",
        "REGULATION OF ",
        "PROCESS",
        "PATHWAY",
        "METABOLIC PROCESS"
    ]

    for w in remove_words:
        name = name.replace(w, "")

    name = name.strip()

    name = "\n".join(textwrap.wrap(name, 20))

    return name


def pathway_heatmap(pathways, pathway_attn, dataset):

    arr = pathway_attn.detach().cpu().numpy()
    meta = dataset.meta.copy()

    importance = arr.std(axis=0) * arr.mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:20]

    mat = arr[:, top_idx]

    names = [clean_pathway_name(pathways[i]) for i in top_idx]

    order = np.lexsort((meta["y_reg"].values, meta["y_cls"].values))
    mat = mat[order, :]

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        pd.DataFrame(mat, columns=names),
        cmap="viridis",
        yticklabels=False
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)

    plt.title("Top Pathway Attention Across Samples")

    plt.tight_layout()

    plt.savefig(f"{OUTDIR}/pathway_heatmap.png", dpi=300)

    plt.close()



def cell_umap(cell_emb, dataset, pred_reg):
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.3, random_state=0)
    z = reducer.fit_transform(cell_emb.detach().cpu().numpy())

    meta = dataset.meta.copy()

    color_map = {
        "primary": "#31bbab",
        "non_primary": "#d55231"
    }
    colors = meta["label_name"].map(color_map).fillna("#999999")

    plt.figure(figsize=(7, 6))
    plt.scatter(
        z[:, 0],
        z[:, 1],
        c=colors,
        s=100,
        edgecolor="black",
        linewidth=0.5
    )

    # overlay predicted proximity as text for primary
    pred = pred_reg.detach().cpu().numpy()
    for i, s in enumerate(dataset.sample_names):
        if meta.iloc[i]["label_name"] == "primary":
            plt.text(z[i, 0], z[i, 1], f"{pred[i]:.2f}", fontsize=8)

    for label, color in color_map.items():
        plt.scatter([], [], c=color, label=label, s=90, edgecolor="black")

    plt.legend(title="Class")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("PGNN Cell Embedding")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/cell_umap.png", dpi=300)
    plt.close()


def engineered_vs_primary_genes(dataset, gene_attn):
    expr = pd.read_csv(Config.EXPR_FILE, index_col=0).T
    expr = expr.loc[dataset.sample_names, dataset.gene_order]

    meta = dataset.meta.copy()

    primary = expr.loc[meta["label_name"] == "primary"]
    non_primary = expr.loc[meta["label_name"] == "non_primary"]

    diff = non_primary.mean(axis=0) - primary.mean(axis=0)
    effect = diff.abs()

    arr = gene_attn.detach().cpu().numpy()
    if arr.ndim == 3:
        gene_scores = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        gene_scores = arr.mean(axis=0)
    else:
        gene_scores = arr.flatten()

    df = pd.DataFrame({
        "gene": expr.columns,
        "logFC_non_primary_minus_primary": diff.values,
        "effect_size": effect.values,
        "pgnn_score": gene_scores
    })

    df["combined_score"] = df["effect_size"] * df["pgnn_score"]
    df = df.sort_values("combined_score", ascending=False)

    df.to_csv(f"{OUTDIR}/engineered_vs_primary_genes.csv", index=False)


def plot_distinguish_genes():
    df = pd.read_csv(f"{OUTDIR}/engineered_vs_primary_genes.csv")

    top_up = df.sort_values("logFC_non_primary_minus_primary", ascending=False).head(10)
    top_down = df.sort_values("logFC_non_primary_minus_primary", ascending=True).head(10)
    top = pd.concat([top_up, top_down])

    plt.figure(figsize=(7, 7))
    sns.barplot(
        data=top,
        x="logFC_non_primary_minus_primary",
        y="gene",
        hue="gene",
        palette="RdBu_r",
        legend=False
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Non-primary − Primary mean expression")
    plt.title("Genes Distinguishing Non-primary vs Primary Hepatocytes")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/distinguish_genes.png", dpi=300)
    plt.close()


def plot_architecture():
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(16, 4))
    boxes = [
        "Gene Expression\n(filtered genes)",
        "Per-gene\nEmbedding",
        "Gene → Pathway\nAttention",
        "Pathway Graph\nNetwork",
        "Pathway → Cell\nAttention",
        "Cell Embedding",
        "Cls + Proximity\nPrediction"
    ]
    x = np.linspace(0.05, 0.85, len(boxes))

    for i, b in enumerate(boxes):
        rect = patches.FancyBboxPatch(
            (x[i], 0.45), 0.12, 0.18,
            boxstyle="round,pad=0.02",
            fc="#82c0dc",
            ec="black"
        )
        ax.add_patch(rect)
        ax.text(x[i] + 0.06, 0.54, b, ha="center", va="center", fontsize=10)

        if i < len(boxes) - 1:
            ax.arrow(x[i] + 0.12, 0.54, 0.03, 0,
                     head_width=0.02, length_includes_head=True)

    ax.axis("off")
    plt.title("PGNN v5 Architecture", fontsize=14)
    plt.savefig(f"{OUTDIR}/model_architecture.png", dpi=300)
    plt.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, dataset, pathways = load_model(device)
    pred_cls, pred_reg, cell, pw, gene_attn, pathway_attn = extract_embeddings(
        model, dataset, device
    )

    plot_top_pathways(pathways, pathway_attn)
    top_genes_global(dataset, gene_attn)
    pathway_heatmap(pathways, pathway_attn, dataset)
    cell_umap(cell, dataset, pred_reg)
    engineered_vs_primary_genes(dataset, gene_attn)
    plot_distinguish_genes()
    plot_architecture()


if __name__ == "__main__":
    main()