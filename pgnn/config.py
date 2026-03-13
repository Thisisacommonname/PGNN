# pgnn/config.py
import os


class Config:
    #paths setting
    GMT_FILE = "data/Liver_pathways_hepatocyte_filtered.gmt"
    EXPR_FILE = "data/normalized_expression_matrix.csv"
    META_FILE = "data/RNAseq_metadata_for_PGNN.csv"

    SAVE_DIR = "results/models/v5_primary_proximity"
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(SAVE_DIR, "pgnn_best.pt")

    ANALYSIS_DIR = "results/analysis/v5_primary_proximity"
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    #feature selection, really low number of samples
    TOP_VAR_GENES = 1500

    #model light run due to limited primary samples
    GENE_DIM = 16
    PATHWAY_DIM = 16
    PW_GNN_LAYERS = 2
    DROPOUT = 0.30
    PW_GRAPH_JACCARD = 0.08

    #training settings
    BATCH_SIZE = 4
    EPOCHS = 150
    LR = 3e-4
    WEIGHT_DECAY = 5e-4
    PATIENCE = 25
    RANDOM_SEED = 42

    
    # regression = primary / fidelity
    LAMBDA_REG = 1.0

    #classification, primary vs. non-primary
    LAMBDA_CLS = 0.2

    #label 
    PRIMARY_GROUPS = {"Primary_hepatocytes"}
    EXCLUDE_GROUPS = set()  