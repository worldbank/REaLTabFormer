import random
import pandas as pd
import pytest

from realtabformer import data_utils as du

df = pd.DataFrame({
    "float": [1.0, 0.123, 32.2334, 100.32],
    "int": [1, 23, 4, 1232],
    "datetime": [
        pd.Timestamp("20180310"),
        pd.Timestamp("20190310"),
        pd.Timestamp("20200316"),
        pd.Timestamp("20181110")],
    "string": [
        "deep", "learning", "data", "science"]})


def test_SPECIAL_COL_SEP():
    # Make sure that any changes in SPECIAL_COL_SEP
    # is deliberate. So have this test here.
    assert du.SPECIAL_COL_SEP == "___"


def test_ColDataType():
    assert du.ColDataType.types() == ["NUMERIC", "DATETIME", "CATEGORICAL"]


def test_SpecialTokens():
    # The order of the tokens as defined
    # in SpecialTokens matters, so
    # we need to make sure that they
    # don't change accidentaly!
    assert du.SpecialTokens.tokens() == [
        "[UNK]",
        "[SEP]",
        "[PAD]",
        "[CLS]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
        "[BMEM]",
        "[EMEM]",
        "[RMASK]",
        "[SPTYPE]",
    ]


def test_get_input_ids():
    # df = du.process_data(df)
    ddf = df.copy()

    with pytest.raises(AssertionError):
        vocab = du.build_vocab(ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True)

    ddf.columns = [f"{idx}___{dtype}___{col}" for idx, (col, dtype) in enumerate(zip(ddf.columns, ["NUMERIC", "NUMERIC", "DATETIME", "CATEGORICAL"]))]
    vocab = du.build_vocab(ddf.astype(str), special_tokens=du.SpecialTokens.tokens(), add_columns=True)

    example = {
        "0___NUMERIC___float": 32.32, "1___NUMERIC___int": 13,
        "2___DATETIME___datetime": pd.Timestamp("20180310"),
        "3___CATEGORICAL___string": "learning"}

    with pytest.raises(AssertionError):
        # Raise assertion for return_token_type_ids=True
        out = du.get_input_ids(
            example,
            vocab=vocab,
            columns=ddf.columns,
            mask_rate=0,
            return_label_ids=True,
            return_token_type_ids=True,
            affix_bos=True,
            affix_eos=True
        )

    out = du.get_input_ids(
        example,
        vocab=vocab,
        columns=ddf.columns,
        mask_rate=0,
        return_label_ids=True,
        return_token_type_ids=False,
        affix_bos=True,
        affix_eos=True
    )

    assert "label_ids" in out
    assert out["label_ids"] == out["input_ids"]
    assert out["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert out["input_ids"][-1] == vocab["token2id"][du.SpecialTokens.EOS]

    out = du.get_input_ids(
        example,
        vocab=vocab,
        columns=ddf.columns,
        mask_rate=0,
        return_label_ids=True,
        return_token_type_ids=False,
        affix_bos=True,
        affix_eos=False
    )

    assert "label_ids" in out
    assert out["label_ids"] == out["input_ids"]
    assert out["input_ids"][0] == vocab["token2id"][du.SpecialTokens.BOS]
    assert out["input_ids"][-1] != vocab["token2id"][du.SpecialTokens.EOS]


def test_build_vocab_no_add_columns():
    # Test vocab without special tokens.
    # Convert all data to string for convenience.
    vocab = du.build_vocab(df.astype(str), add_columns=False)
    # Check vocab size
    assert len(vocab["id2token"]) == 16
    assert len(vocab["token2id"]) == 16

    # Check id range
    assert min(vocab["id2token"]) == 0
    assert max(vocab["id2token"]) == 15

    # Check that the token2id and id2token
    # actually are inverse maps of each other.
    for token, t_id in vocab["token2id"].items():
        assert vocab["token2id"][token] == t_id

    # Check values
    assert vocab["id2token"][0] == "0.123"
    assert vocab["id2token"][1] == "1.0"

    # Note that "100.32" goes before "32.2334"
    # because we changed the dtype to string.
    # The sorting applies to the string-converted
    # data.
    assert vocab["id2token"][2] == "100.32"
    assert vocab["id2token"][3] == "32.2334"

    # Sample other tokens.
    assert vocab["id2token"][5] == "1232"
    assert vocab["id2token"][10] == "2019-03-10"
    assert vocab["id2token"][14] == "learning"

    # Check the column token ids output.
    assert vocab["column_token_ids"]["int"] == [4, 5, 6, 7]
    assert vocab["column_token_ids"]["string"] == [12, 13, 14, 15]


def test_build_vocab_add_columns():
    # Test vocab without special tokens.
    # Convert all data to string for convenience.
    ddf = df.copy()
    ddf.columns = [f"{idx}___{dtype}___{col}" for idx, (col, dtype) in enumerate(zip(ddf.columns, ["NUMERIC", "NUMERIC", "DATETIME", "CATEGORICAL"]))]

    vocab = du.build_vocab(ddf.astype(str), add_columns=True)
    # Check vocab size
    assert len(vocab["id2token"]) == (16 + 4)
    assert len(vocab["token2id"]) == (16 + 4)

    # Check id range
    assert min(vocab["id2token"]) == 0
    assert max(vocab["id2token"]) == 19

    # Check column values
    assert vocab["id2token"][16] == "0___NUMERIC"
    assert vocab["id2token"][17] == "1___NUMERIC"
    assert vocab["id2token"][18] == "2___DATETIME"
    assert vocab["id2token"][19] == "3___CATEGORICAL"


def test_process_numeric_data():
    series, transform_data = du.process_numeric_data(
        df["int"], max_len=10, numeric_precision=4)
    expected_out = ["0001", "0023", "0004", "1232"]

    for v, e in zip(series, expected_out):
        assert v == e

    # Make sure that max_len doesn't truncate integral
    # data.
    series, transform_data = du.process_numeric_data(
        df["int"], max_len=2, numeric_precision=4)

    expected_out = ["0001", "0023", "0004", "1232"]

    for v, e in zip(series, expected_out):
        assert v == e

    series, transform_data = du.process_numeric_data(
        df["float"], max_len=5, numeric_precision=4)

    # Note that the value of 0.123 will be truncated
    # to 000.1 because we prioritize the padding at
    # the leading digits of the values before truncating
    # the max length.
    expected_out = ["001.0", "000.1", "032.2", "100.3"]

    for v, e in zip(series, expected_out):
        assert v == e

    # Check that the processing of the
    # data is sensitive to the resolution loss.
    with pytest.raises(AssertionError):
        # This should raise an AssertionError because the
        # desired max_len of 4 will generate a loss in
        # the numeric resolution of the data
        series, transform_data = du.process_numeric_data(
            df["float"], max_len=4, numeric_precision=4)


def test_process_data():
    pr_df, _ = du.process_data(
        df, numeric_max_len=10, numeric_precision=4, numeric_nparts=2)

    # Validate the processed columns
    assert pr_df.shape[1] == 12
    assert pr_df.columns.str.startswith(f"0___{du.ColDataType.NUMERIC}___float").sum() == 4
    assert pr_df.columns.str.startswith(f"1___{du.ColDataType.NUMERIC}___int").sum() == 2
    assert pr_df.columns.str.startswith(f"2___{du.ColDataType.DATETIME}___datetime").sum() == 5
    assert pr_df.columns.str.startswith(f"3___{du.ColDataType.CATEGORICAL}___string").sum() == 1

    # Validate that the columns are properly ordered (default)
    start_idx = 0
    for col in pr_df.columns:
        col_idx = int(col.split(du.SPECIAL_COL_SEP)[0])
        if col_idx != start_idx:
            assert (start_idx + 1) == col_idx
            start_idx = col_idx

    for col in pr_df.columns:
        assert pr_df[col].str.startswith(col).all()


@pytest.mark.parametrize("first_col_type", [None, du.ColDataType.CATEGORICAL, du.ColDataType.NUMERIC])
def test_process_data_first_col_type(first_col_type):
    # Test for categorical first cols
    pr_df, _ = du.process_data(
        df, numeric_max_len=10, numeric_precision=4, numeric_nparts=2,
        first_col_type=first_col_type
    )

    start_idx = 0
    seen_last = False
    for idx, col in enumerate(pr_df.columns):
        if first_col_type is not None:
            if idx == 0:
                # Make sure that our set first_col_type
                # is actually the first column type in the
                # returned data.
                assert first_col_type in col
            elif first_col_type not in col:
                seen_last = True

            if not seen_last:
                if first_col_type == du.ColDataType.CATEGORICAL:
                    assert first_col_type in col
                else:
                    # NUMERIC and DATETIME fall under the same
                    # general numeric category.
                    assert (first_col_type in col) or (du.ColDataType.DATETIME in col)
            else:
                assert first_col_type not in col
        else:
            # If no preferred first_col_type is set,
            # we use the actual order of the input
            # dataframe.
            col_idx = int(col.split(du.SPECIAL_COL_SEP)[0])
            if col_idx != start_idx:
                assert (start_idx + 1) == col_idx
                start_idx = col_idx


def test_decode_processed_column():
    assert du.decode_processed_column(f"0___{du.ColDataType.NUMERIC}___float_00") == "float_00"
    assert du.decode_processed_column(f"0___{du.ColDataType.NUMERIC}___int_01") == "int_01"
    assert du.decode_processed_column(f"0___{du.ColDataType.DATETIME}___datetime_03") == "datetime_03"
    assert du.decode_processed_column(f"0___{du.ColDataType.CATEGORICAL}___string_02") == "string_02"

    # Check if leading numeric prefix is long.
    assert du.decode_processed_column(f"121___{du.ColDataType.CATEGORICAL}___string_02") == "string_02"

    # Check if columns is not in the expected format.
    assert du.decode_processed_column("random_col") == "random_col"


@pytest.mark.parametrize("dtype", du.ColDataType.types() + ["SOMEDTYPE"])
def test_encode_processed_column(dtype):
    idx = 0
    col = "foo"

    if dtype != "SOMEDTYPE":
        # Expected form: "0___(NUMERIC|DATETIME|CATEGORICAL)___foo"
        out = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}"
        assert du.encode_processed_column(idx, dtype, col) == out
    else:
        with pytest.raises(AssertionError):
            du.encode_processed_column(idx, dtype, col)


@pytest.mark.parametrize("dtype", du.ColDataType.types() + ["SOMEDTYPE"])
def test_extract_processed_column(dtype):
    col = "foo012_%$@&#"

    for _ in range(10):
        idx = random.randint(0, 1000)

        if dtype != "SOMEDTYPE":
            # Expected form: "0___(NUMERIC|DATETIME|CATEGORICAL)___foo012_%$@&#"
            expected = f"{idx}{du.SPECIAL_COL_SEP}{dtype}"
            inp = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}{du.SPECIAL_COL_SEP}00"
            assert du.extract_processed_column(inp) == expected
        else:
            inp = f"{idx}{du.SPECIAL_COL_SEP}{dtype}{du.SPECIAL_COL_SEP}{col}{du.SPECIAL_COL_SEP}00"
            assert du.extract_processed_column(inp) is None
