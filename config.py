import os
import random
import numpy as np
import torch

# Random seed and reproducibility settings
RANDOM_SEED = 0  # Options: 0, 12345, 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Classes and phases
CLASSES = ["ADHD", "MDD"]
NUM_CLASSES = len(CLASSES)
PHASES = ["train", "valid", "test"]

# Age groups and groups
AGE_GROUPS = [f"ageGroup_{i}" for i in range(2)]
GROUPS = [(age_group, label) for age_group in AGE_GROUPS for label in CLASSES]

# Paths to your data (update these paths as needed)
INPUT_PATH_PARQUET = '/path/to/new_td_raw_band_0.5_50'
METADATA_PATH = '/path/to/TDBRAIN_participants_V2.xlsx'
SAVE_PATH = '/path/to/runs'

# k-fold settings
N_SPLITS = 4

# Data parameters
CHANNEL_NUM = 26
EEG_TASK = ["EC", "EO"]
LABEL_BALANCED_SAMPLING = True

# Mixup settings
USE_MIXUP = False
MIXUP_ALPHA = 0.2

# Transformer hyperparameters
WINDOW = 32
STRIDE = WINDOW // 2
RAW_SEQUENCE_LENGTH = 512
SEQUENCE_LENGTH = int((RAW_SEQUENCE_LENGTH - WINDOW) / STRIDE) + 2  # plus CLS token
NUM_LAYERS = 5
EMBED_DIM = 128
HEAD_SIZE = 8
NUM_HEADS = EMBED_DIM // HEAD_SIZE
FF_DIM = EMBED_DIM * 4
DROPOUT_RATE = 0.1

# Training parameters
EPOCH_STEPS = 500
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-2
PATIENCE = 100
WORST_GROUP_MONITORING = True
EMA_ALPHA = 0.3

# DRO and DFR settings
APPLY_GROUP_DRO = False
STEP_SIZE = 1e-2
DFR = False
NUM_RETRAINS = 20
DFR_RETRAIN = False

if DFR:
    EPOCHS = 300
    PATIENCE = 300
    LABEL_BALANCED_SAMPLING = True

# Debiased Batch Normalization parameters
DEBIASED_BN = True
SUB_BATCH_NUM = 8
SUB_BATCH_SIZE = 8

if BATCH_SIZE % SUB_BATCH_NUM != 0:
    raise ValueError(f"Batch size must be divisible by {SUB_BATCH_NUM}")

ERM = not (DFR or APPLY_GROUP_DRO)
