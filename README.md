# PGNN

Pathway Guided Neural Network for evaluating hepatocyte fidelity using transcriptomic data。

## Overview

PGNN integrates biological pathway knowledge into neural network architecture to evaluate engineered hepatocyte transcriptomes.

## Input

- RNAseq expression matrix
- Sample metadata
- Liver pathway gene sets

## Training

Run:

python pgnn/train.py

## Visualization

Run:

python pgnn/interpret_pgnn_v2.py

## Model

Pretrained model:

models/pgnn_v5_best.pt
