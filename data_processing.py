import os
import re
import random
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold
import torch

from config import CLASSES, AGE_GROUPS, PHASES, LABEL_BALANCED_SAMPLING, RAW_SEQUENCE_LENGTH, BATCH_SIZE, EEG_TASK, RANDOM_SEED

def load_metadata(filepath, all_files_list):
    """
    Load and filter metadata from an Excel file based on available EEG files and target classes.

    Args:
        filepath (str): Path to the Excel file.
        all_files_list (list): List of available EEG file names.

    Returns:
        DataFrame: Filtered metadata DataFrame.
    """
    df = pd.read_excel(filepath, engine="openpyxl")
    # Filter subjects with at least one session file available
    df = df[df['participants_ID'].apply(
        lambda id: any(f"{id}_ses-{sess}_task-restEC.parquet" in all_files_list for sess in range(1, 4))
    )]
    df = df[df["indication"].isin(CLASSES)]
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    nan_counts = df['age'].isna().groupby(df["indication"]).sum().astype(int)
    for class_status, count in nan_counts.items():
        if count > 0:
            print(f"Dropped {count} rows with NaN ages from class '{class_status}'")
    df = df.dropna(subset=['age'])
    return df

def split_data(metadata_df, all_files_list, fold_num, n_splits):
    """
    Split metadata into train, validation, and test sets and create session paths.

    Args:
        metadata_df (DataFrame): Metadata DataFrame.
        all_files_list (list): List of available session file names.
        fold_num (int): Current fold number.
        n_splits (int): Number of splits for KFold.

    Returns:
        tuple: (fold_ids, fold_session_paths)
    """
    fold_ids = {phase: {age_group: {label: [] for label in CLASSES} for age_group in AGE_GROUPS} for phase in PHASES}
    fold_session_paths = {phase: {age_group: {label: [] for label in CLASSES} for age_group in AGE_GROUPS} for phase in PHASES}

    for label in CLASSES:
        label_df = metadata_df[metadata_df["indication"].str.contains(re.escape(label), case=False, na=False)]
        ids_ages = label_df[['participants_ID', 'age']].drop_duplicates(subset='participants_ID')
        ids_ages = ids_ages.sort_values('age')
        ids_ages['age_group'] = pd.qcut(ids_ages['age'], q=2, labels=AGE_GROUPS)
        for age_group in AGE_GROUPS:
            ids_list = ids_ages[ids_ages['age_group'] == age_group]['participants_ID'].unique().tolist()
            age_group_kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
            train_val_idx, test_idx = list(age_group_kf.split(ids_list))[fold_num]
            train_idx, val_idx = train_test_split(train_val_idx, test_size=1/3, random_state=RANDOM_SEED)
            fold_ids['train'][age_group][label] = np.array(ids_list)[train_idx].tolist()
            fold_ids['valid'][age_group][label] = np.array(ids_list)[val_idx].tolist()
            fold_ids['test'][age_group][label] = np.array(ids_list)[test_idx].tolist()
            sessions_conditions = list(product(range(1, 4), EEG_TASK))
            for phase in PHASES:
                for sub_id in fold_ids[phase][age_group][label]:
                    for sess, cond in sessions_conditions:
                        file_name = f"{sub_id}_ses-{sess}_task-rest{cond}.parquet"
                        if file_name in all_files_list:
                            fold_session_paths[phase][age_group][label].append(file_name)
    return fold_ids, fold_session_paths

def normalize_data(fold_session_paths, input_path_parquet):
    """
    Load and normalize data frames from session paths.

    Args:
        fold_session_paths (dict): Session paths organized by phase, age group, and label.
        input_path_parquet (str): Directory containing parquet files.

    Returns:
        tuple: (fold_dfs, global_mean, global_std)
    """
    all_train_dfs = []
    for age_group in AGE_GROUPS:
        for label in CLASSES:
            for session_path in fold_session_paths["train"][age_group][label]:
                df = pd.read_parquet(os.path.join(input_path_parquet, session_path))
                all_train_dfs.append(df)
    combined_train_df = pd.concat(all_train_dfs)
    global_mean = combined_train_df.mean()
    global_std = combined_train_df.std()
    fold_dfs = {phase: {age_group: {label: [] for label in CLASSES} for age_group in AGE_GROUPS} for phase in PHASES}
    for phase in PHASES:
        for age_group in AGE_GROUPS:
            for label in CLASSES:
                for session_path in fold_session_paths[phase][age_group][label]:
                    df = pd.read_parquet(os.path.join(input_path_parquet, session_path))
                    df = (df - global_mean) / global_std
                    fold_dfs[phase][age_group][label].append(df)
    return fold_dfs, global_mean, global_std

def process_df(df, start_idx, sequence_length):
    """
    Process a DataFrame to extract a slice.

    Args:
        df (DataFrame): Input data.
        start_idx (int): Starting index.
        sequence_length (int): Length of sequence to extract.

    Returns:
        np.ndarray: Transposed slice of data.
    """
    slice_df = df.iloc[start_idx : start_idx + sequence_length]
    return slice_df.to_numpy().T

def last_axis_normalization(data):
    """
    Normalize data along the last axis.

    Args:
        data (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized data.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True) + 1e-6
    return (data - mean) / std

def apply_mixup(x, y, g, alpha, num_classes):
    """
    Apply Mixup augmentation on batch data.

    Args:
        x (Tensor): Input tensor.
        y (Tensor): Labels.
        g (Tensor): Group labels.
        alpha (float): Mixup alpha parameter.
        num_classes (int): Number of classes.

    Returns:
        Tuple: Mixed inputs, labels, and group labels.
    """
    batch_size = x.size(0)
    device = x.device
    lam = torch.tensor(np.random.beta(alpha, alpha, batch_size) if alpha > 0 else np.ones(batch_size),
                         device=device, dtype=x.dtype)
    lam_x = lam.view(batch_size, *([1] * (x.dim() - 1)))
    candidate_mask = y.unsqueeze(1) != y.unsqueeze(0)
    random_scores = torch.rand(batch_size, batch_size, device=device)
    random_scores = random_scores.masked_fill(~candidate_mask, -1)
    indices = random_scores.argmax(dim=1)
    valid = candidate_mask.any(dim=1)
    if (~valid).any():
        fallback_indices = torch.randint(0, batch_size, ((~valid).sum(),), device=device)
        indices[~valid] = fallback_indices
    mixed_x = lam_x * x + (1 - lam_x) * x[indices]
    lam_y = lam.view(batch_size, 1)
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    y_partner = torch.nn.functional.one_hot(y[indices], num_classes=num_classes).float()
    mixed_y = lam_y * y_one_hot + (1 - lam_y) * y_partner
    mask = (lam >= 0.5).view(batch_size, *([1] * (g.dim() - 1)))
    mixed_g = torch.where(mask, g, g[indices])
    return mixed_x, mixed_y, mixed_g

def batch_generator(dfs, device, training=False, batch_size=BATCH_SIZE, sequence_length=RAW_SEQUENCE_LENGTH,
                    alpha=MIXUP_ALPHA, use_mixup=False, num_classes=2):
    """
    Generator that yields batches for training/evaluation.

    Args:
        dfs (dict): DataFrames organized by age group and label.
        device (torch.device): Computation device.
        training (bool): Whether in training mode.
        batch_size (int): Batch size.
        sequence_length (int): Sequence length.
        alpha (float): Mixup alpha parameter.
        use_mixup (bool): Flag to apply Mixup.
        num_classes (int): Number of classes.

    Yields:
        Tuple: (input_tensor, label_tensor, group_tensor)
    """
    groups = [(age_group, label) for age_group in AGE_GROUPS for label in CLASSES]
    if LABEL_BALANCED_SAMPLING:
        sampling_probs = [1 / len(groups)] * len(groups)
    else:
        weights = [len(dfs[age_group][label]) for age_group, label in groups]
        total_weight = sum(weights)
        sampling_probs = [w / total_weight for w in weights]
    while True:
        input_data = []
        label_data = []
        age_group_data = []
        random_picks = random.choices(groups, weights=sampling_probs, k=batch_size)
        for age_group, label in random_picks:
            df = random.choice(dfs[age_group][label])
            start_idx = np.random.randint(0, len(df) - sequence_length)
            sliced_data = process_df(df, start_idx, sequence_length)
            input_data.append(sliced_data)
            label_data.append(CLASSES.index(label))
            age_group_data.append(AGE_GROUPS.index(age_group))
        input_tensor = torch.tensor(np.stack(input_data), dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.long)
        group_tensor = torch.tensor(age_group_data, dtype=torch.long)
        if training and use_mixup:
            input_tensor, label_tensor, group_tensor = apply_mixup(input_tensor, label_tensor, group_tensor, alpha, num_classes)
        yield input_tensor.to(device), label_tensor.to(device), group_tensor.to(device)

def eval_generator(df, device, sequence_length, eval_stride=16):
    """
    Generate evaluation batches from a DataFrame.

    Args:
        df (DataFrame): Input DataFrame.
        device (torch.device): Computation device.
        sequence_length (int): Sequence length.
        eval_stride (int): Stride for evaluation.

    Returns:
        Tensor: Evaluation batch tensor.
    """
    input_data = []
    for start_idx in range(0, len(df) - sequence_length, sequence_length // eval_stride):
        sliced_data = process_df(df, start_idx, sequence_length)
        input_data.append(sliced_data)
    return torch.tensor(np.stack(input_data), dtype=torch.float32).to(device)
