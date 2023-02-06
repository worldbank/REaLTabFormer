from pathlib import Path

import pytest
from transformers import EncoderDecoderConfig
from transformers.models.gpt2 import GPT2Config

import realtabformer
from realtabformer.realtabformer import ModelType, REaLTabFormer

RANDOM_SEED = 1029


def test_ModelType():
    assert ModelType.types() == ["tabular", "relational"]


def test_default_init():
    model_types = [ModelType.tabular, ModelType.relational]

    for model_type in model_types:
        rtf_model = REaLTabFormer(model_type)

        # Track the variables that we have tested to
        # make sure that all variables that will be
        # added or removed in the future will be caught
        # by the test.
        model_vars_tested = set()

        assert rtf_model.model_type == model_type
        model_vars_tested.add("model_type")

        # Check default dir arguments
        assert isinstance(rtf_model.checkpoints_dir, Path)
        assert rtf_model.checkpoints_dir.name == "rtf_checkpoints"
        model_vars_tested.add("checkpoints_dir")

        assert isinstance(rtf_model.samples_save_dir, Path)
        assert rtf_model.samples_save_dir.name == "rtf_samples"
        model_vars_tested.add("samples_save_dir")

        assert rtf_model.epochs == 100
        model_vars_tested.add("epochs")

        assert rtf_model.batch_size == 8
        model_vars_tested.add("batch_size")

        assert rtf_model.random_state == 1029
        model_vars_tested.add("random_state")

        assert rtf_model.train_size == 1
        model_vars_tested.add("train_size")

        assert rtf_model.early_stopping_patience == 5
        model_vars_tested.add("early_stopping_patience")

        assert rtf_model.early_stopping_threshold == 0
        model_vars_tested.add("early_stopping_threshold")

        assert rtf_model.mask_rate == 0
        model_vars_tested.add("mask_rate")

        assert rtf_model.numeric_nparts == 1
        model_vars_tested.add("numeric_nparts")

        assert rtf_model.numeric_precision == 4
        model_vars_tested.add("numeric_precision")

        assert rtf_model.numeric_max_len == 10
        model_vars_tested.add("numeric_max_len")

        if model_type == ModelType.tabular:
            with pytest.raises(AttributeError):
                # The argument `output_max_length` is not set
                # for the tabular model.
                assert rtf_model.output_max_length

            # Implicitly tests `_init_tabular`
            assert isinstance(rtf_model.tabular_config, GPT2Config)
            assert rtf_model.tabular_config.n_layer == 6
            model_vars_tested.add("tabular_config")
        else:
            assert rtf_model.output_max_length == 512
            model_vars_tested.add("output_max_length")

            assert rtf_model.freeze_parent_model
            model_vars_tested.add("freeze_parent_model")

            # Relational model
            assert rtf_model.parent_vocab is None
            model_vars_tested.add("parent_vocab")

            assert rtf_model.parent_gpt2_config is None
            model_vars_tested.add("parent_gpt2_config")

            assert rtf_model.parent_gpt2_state_dict is None
            model_vars_tested.add("parent_gpt2_state_dict")

            assert rtf_model.parent_col_transform_data is None
            model_vars_tested.add("parent_col_transform_data")

            # Implicitly tests `_init_relational`
            assert isinstance(rtf_model.relational_config, EncoderDecoderConfig)
            assert isinstance(rtf_model.relational_config.encoder, GPT2Config)
            assert isinstance(rtf_model.relational_config.decoder, GPT2Config)
            assert rtf_model.relational_config.encoder.n_layer == 6
            assert rtf_model.relational_config.decoder.n_layer == 6

            model_vars_tested.add("relational_config")

        # Validate the implicit default values in `training_args_kwargs`
        assert rtf_model.training_args_kwargs["evaluation_strategy"] == "steps"
        assert (
            rtf_model.training_args_kwargs["output_dir"]
            == rtf_model.checkpoints_dir.as_posix()
        )
        assert rtf_model.training_args_kwargs["evaluation_strategy"] == "steps"

        assert rtf_model.training_args_kwargs["metric_for_best_model"] == "loss"
        assert rtf_model.training_args_kwargs["overwrite_output_dir"] is True
        assert rtf_model.training_args_kwargs["num_train_epochs"] == rtf_model.epochs
        assert (
            rtf_model.training_args_kwargs["per_device_train_batch_size"]
            == rtf_model.batch_size
        )
        assert (
            rtf_model.training_args_kwargs["per_device_eval_batch_size"]
            == rtf_model.batch_size
        )

        assert rtf_model.training_args_kwargs["gradient_accumulation_steps"] == 4
        assert rtf_model.training_args_kwargs["remove_unused_columns"] is True
        assert rtf_model.training_args_kwargs["logging_steps"] == 100
        assert rtf_model.training_args_kwargs["save_steps"] == 100
        assert rtf_model.training_args_kwargs["eval_steps"] == 100
        assert rtf_model.training_args_kwargs["load_best_model_at_end"] is True
        assert (
            rtf_model.training_args_kwargs["save_total_limit"]
            == rtf_model.early_stopping_patience + 1
        )
        model_vars_tested.add("training_args_kwargs")

        # Validate empty-initialized attributes
        list_defaults = [
            "columns",
            "drop_na_cols",
            "processed_columns",
            "numeric_columns",
            "datetime_columns",
        ]
        for ld in list_defaults:
            assert (
                isinstance(getattr(rtf_model, ld), list)
                and len(getattr(rtf_model, ld)) == 0
            )

        dict_defaults = ["column_dtypes", "column_has_missing", "vocab", "col_idx_ids"]
        for dd in dict_defaults:
            assert (
                isinstance(getattr(rtf_model, dd), dict)
                and len(getattr(rtf_model, dd)) == 0
            )

        none_defaults = [
            "model",
            "tabular_max_length",
            "relational_max_length",
            "tabular_col_size",
            "relational_col_size",
            "experiment_id",
            "col_transform_data",
            "in_col_transform_data",
            "target_col",
            "trainer_state",
        ]
        for nd in none_defaults:
            assert getattr(rtf_model, nd) is None

        assert rtf_model.realtabformer_version == realtabformer.__version__
        model_vars_tested.add("realtabformer_version")

        model_vars_tested.update(list_defaults)
        model_vars_tested.update(dict_defaults)
        model_vars_tested.update(none_defaults)

        model_vars = set(vars(rtf_model))

        print(model_vars.difference(model_vars_tested))

        assert len(model_vars.difference(model_vars_tested)) == 0
        assert len(model_vars) == len(model_vars_tested), f"{model_vars}...{model_vars_tested}"


# def test_tabular_init():
#     training_args_kwargs = dict(
#         logging_steps=100,
#         save_steps=100,
#         eval_steps=100,
#         save_total_limit=10,
#         gradient_accumulation_steps=4,
#     )

#     samples_save_dir = "samples_save_dir"
#     batch_size = 8
#     epochs = 10
#     mask_rate = 0.2
#     train_size = 1

#     tabular_rtf = REaLTabFormer(
#         model_type="tabular",
#         samples_save_dir=samples_save_dir,
#         epochs=epochs, batch_size=batch_size,
#         mask_rate=mask_rate,
#         train_size=train_size,
#         random_state=RANDOM_SEED,
#         **training_args_kwargs)
