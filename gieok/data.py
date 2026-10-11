"""Data loading and preprocessing for GIEOK."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load data manifest CSV.

    Expected columns: id, speaker_id, features, label, language, split

    Args:
        manifest_path: Path to manifest CSV

    Returns:
        DataFrame with data manifest
    """
    df = pd.read_csv(manifest_path)

    # Validate required columns
    required_cols = ['id', 'speaker_id', 'features', 'label', 'language', 'split']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in manifest: {missing}")

    return df


def load_cues(cues_path: str) -> Dict[str, List[float]]:
    """
    Load clinical cues CSV.

    Args:
        cues_path: Path to cues CSV

    Returns:
        Dictionary mapping utterance IDs to cue values
    """
    df = pd.read_csv(cues_path)

    # Convert to dictionary
    cues = {}
    for _, row in df.iterrows():
        utterance_id = row['id']
        cue_values = {k: v for k, v in row.items() if k != 'id'}
        cues[utterance_id] = cue_values

    return cues


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    stratify_by: Optional[str] = None,
    speaker_independent: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test splits from manifest.

    Args:
        df: Data manifest DataFrame
        train_ratio: Fraction for training
        stratify_by: Column to stratify by
        speaker_independent: Ensure speakers don't overlap between splits

    Returns:
        (train_df, test_df)
    """
    train_data = []
    test_data = []

    if speaker_independent:
        # Split by speaker
        speakers = df['speaker_id'].unique()
        n_train = int(len(speakers) * train_ratio)
        train_speakers = np.random.choice(speakers, n_train, replace=False)

        for _, row in df.iterrows():
            if row['speaker_id'] in train_speakers:
                train_data.append(row)
            else:
                test_data.append(row)
    else:
        # Random split
        n_train = int(len(df) * train_ratio)
        indices = np.random.permutation(len(df))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_data = [df.iloc[i] for i in train_indices]
        test_data = [df.iloc[i] for i in test_indices]

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


def load_features(features_path: str) -> np.ndarray:
    """
    Load precomputed speech features.

    Args:
        features_path: Path to feature file

    Returns:
        Feature array
    """
    # Support common formats
    if features_path.endswith('.npy'):
        return np.load(features_path)
    elif features_path.endswith('.npz'):
        data = np.load(features_path)
        return data['features']
    elif features_path.endswith('.pt'):
        import torch
        return torch.load(features_path).numpy()
    else:
        raise ValueError(f"Unsupported format: {features_path}")
