"""This suite tests for the rtf_sampler.py module."""
import torch
import fixtures as fx

from realtabformer.rtf_sampler import TabularSampler


def test_TabularSampler():
    tab_fx = fx.tabular_fixtures(fit=True)
    rtf_model = tab_fx["rtf_model"]
    df = tab_fx["df"]
    n_samples = 100
    gen_batch = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    constrain_tokens_gen = True
    continuous_empty_limit = 10
    suppress_tokens = None

    tabular_sampler = TabularSampler(
        model_type=rtf_model.model_type,
        model=rtf_model.model,
        vocab=rtf_model.vocab,
        processed_columns=rtf_model.processed_columns,
        max_length=rtf_model.tabular_max_length,
        col_size=rtf_model.tabular_col_size,
        col_idx_ids=rtf_model.col_idx_ids,
        columns=rtf_model.columns,
        datetime_columns=rtf_model.datetime_columns,
        column_dtypes=rtf_model.column_dtypes,
        drop_na_cols=rtf_model.drop_na_cols,
        col_transform_data=rtf_model.col_transform_data,
        random_state=rtf_model.random_state,
        device=device,
    )

    pred_df = tabular_sampler.sample_tabular(
        n_samples=n_samples,
        gen_batch=gen_batch,
        device=device,
        constrain_tokens_gen=constrain_tokens_gen,
        continuous_empty_limit=continuous_empty_limit,
        suppress_tokens=suppress_tokens,
    )

    # Check that the columns are ordered in the
    # same way as the input data.
    assert all(df.columns == pred_df.columns)

    # Check that the data types are similar to
    # the training data.
    dtypes = df.dtypes
    pred_dtypes = pred_df.dtypes

    for col in dtypes.index:
        assert dtypes[col] == pred_dtypes[col]
