"""
Adaptive Recommender System for Elderly Cognitive Enhancement
Configuration
"""

class Config:
    # Dataset
    NUM_USERS = 200
    NUM_ITEMS = 100
    NUM_INTERACTIONS = 5000
    NUM_SOCIAL_EDGES = 600
    EMBEDDING_DIM = 128
    CONTENT_EMBED_DIM = 384       # Sentence-BERT output dim

    # Temporal split
    TRAIN_RATIO = 0.8

    # Model
    GAT_HIDDEN_DIM = 64
    GAT_OUTPUT_DIM = 32
    GAT_HEADS = 4
    DROPOUT = 0.3

    # Training
    EPOCHS = 100
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512

    # Evaluation
    TOP_K = [5, 10, 20]

    # QoL feedback loop
    QOL_DIM = 4                   # WHOQOL-BREF domains
    QOL_ALPHA = 0.3               # QoL injection weight

    # User study
    NUM_PARTICIPANTS = 50
    STUDY_WEEKS = 8

    # Paths
    DATA_DIR = "dataset"
    MODEL_DIR = "checkpoints"
    RESULTS_DIR = "results"

    # Reproducibility
    SEED = 42
