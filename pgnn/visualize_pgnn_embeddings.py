import os
import pandas as pd
import umap
import matplotlib.pyplot as plt

EMBED_FILE = "results/cell_embeddings.csv"
META_FILE = "data/RNAseq/RNAseq_metadata_for_PGNN.csv"


#loading embedding
emb = pd.read_csv(EMBED_FILE, index_col=0)
#metadata
meta = pd.read_csv(META_FILE)

meta = meta.set_index("sample")

# merge metadata
data = emb.join(meta)

#run the UMAP
reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.3,
    random_state=42
)

coords = reducer.fit_transform(data.iloc[:, :emb.shape[1]])

data["UMAP1"] = coords[:, 0]
data["UMAP2"] = coords[:, 1]


os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8,6))

groups = data["group"].unique()

for g in groups:

    subset = data[data["group"] == g]

    plt.scatter(
        subset["UMAP1"],
        subset["UMAP2"],
        c=subset["color"],
        s=120,
        label=g,
        edgecolor="black"
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("PGNN Cell Embedding UMAP")

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "results/pgnn_umap_by_sample.png",
    dpi=300
)
