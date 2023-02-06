"""This module contains some common functions that will be
used across the different tests.
"""
from pathlib import Path
import pandas as pd
import torch
from realtabformer import data_utils as du
from realtabformer import REaLTabFormer

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_test_data():
    """Get or load the titanic dataset."""
    data_fname = TEST_DATA_DIR / "titanic.csv"
    data_fname.parent.mkdir(parents=True, exist_ok=True)

    if not data_fname.exists():
        data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", index_col=0)
        data.to_csv(data_fname)
    else:
        data = pd.read_csv(data_fname, index_col=0)

    data = data.drop("Name", axis=1)

    return data


def tabular_fixtures(fit=False):
    """Generates tabular fixtures for the sampler test."""
    df = get_test_data().head(100)

    rtf_model = REaLTabFormer(
        model_type=du.ModelType.tabular,
        epochs=1,
        checkpoints_dir=TEST_DATA_DIR / "rtf_checkpoints",
        samples_save_dir=TEST_DATA_DIR / "rtf_smaples",
    )

    if fit:
        rtf_model.fit(df, device="cuda" if torch.cuda.is_available() else "cpu")

    return dict(
        rtf_model=rtf_model,
        df=df,
    )
