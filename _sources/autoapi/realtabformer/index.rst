:py:mod:`realtabformer`
=======================

.. py:module:: realtabformer


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   __main__/index.rst
   data_utils/index.rst
   realtabformer/index.rst
   rtf_analyze/index.rst
   rtf_datacollator/index.rst
   rtf_exceptions/index.rst
   rtf_sampler/index.rst
   rtf_trainer/index.rst
   rtf_validators/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.REaLTabFormer




.. py:class:: REaLTabFormer(model_type: str, tabular_config: Optional[transformers.models.gpt2.GPT2Config] = None, relational_config: Optional[transformers.EncoderDecoderConfig] = None, parent_realtabformer_path: Optional[pathlib.Path] = None, freeze_parent_model: Optional[bool] = True, checkpoints_dir: str = 'rtf_checkpoints', samples_save_dir: str = 'rtf_samples', epochs: int = 100, batch_size: int = 8, random_state: int = 1029, train_size: float = 1, output_max_length: int = 512, early_stopping_patience: int = 5, early_stopping_threshold: float = 0, mask_rate: float = 0, numeric_nparts: int = 1, numeric_precision: int = 4, numeric_max_len: int = 10, **training_args_kwargs)


   .. py:method:: _invalid_model_type(model_type)


   .. py:method:: _init_tabular(tabular_config)


   .. py:method:: _init_relational(relational_config)


   .. py:method:: _extract_column_info(df: pandas.DataFrame) -> None


   .. py:method:: _generate_vocab(df: pandas.DataFrame) -> dict


   .. py:method:: _check_model()


   .. py:method:: _split_train_eval_dataset(dataset: datasets.Dataset)


   .. py:method:: fit(df: pandas.DataFrame, in_df: Optional[pandas.DataFrame] = None, join_on: Optional[str] = None, resume_from_checkpoint: Union[bool, str] = False, device='cuda', num_bootstrap: int = 500, frac: float = 0.165, frac_max_data: int = 10000, qt_max: Union[str, float] = 0.05, qt_max_default: float = 0.05, qt_interval: int = 100, qt_interval_unique: int = 100, distance: sklearn.metrics.pairwise.manhattan_distances = manhattan_distances, quantile: float = 0.95, n_critic: int = 5, n_critic_stop: int = 2, gen_rounds: int = 3, sensitivity_max_col_nums: int = 20, use_ks: bool = False, full_sensitivity: bool = False, sensitivity_orig_frac_multiple: int = 4, orig_samples_rounds: int = 5, load_from_best_mean_sensitivity: bool = False, target_col: str = None)

      Train the REaLTabFormer model on the tabular data.

      :param df: Pandas DataFrame containing the tabular data that will be generated during sampling.
                 This data goes to the decoder for the relational model.
      :param in_df: Pandas DataFrame containing observations related to `df`, and from which the
                    model will generate data. This data goes to the encoder for the relational model.
      :param join_on: Column name that links the `df` and the `in_df` tables.
      :param resume_from_checkpoint: If True, resumes training from the latest checkpoint in the
                                     checkpoints_dir. If path, resumes the training from the given checkpoint.
      :param device: Device where the model and the training will be run.
                     Use torch devices, e.g., `cpu`, `cuda`, `mps` (experimental)
      :param num_bootstrap: Number of Bootstrap samples
      :param frac: The fraction of the data used for training.
      :param frac_max_data: The maximum number of rows that the training data will have.
      :param qt_max: The maximum quantile for the discriminator.
      :param qt_max_default: The default maximum quantile for the discriminator.
      :param qt_interval: Interval for the quantile check during the training process.
      :param qt_interval_unique: Interval for the quantile check during the training process.
      :param distance: Distance metric used for discriminator.
      :param quantile: The quantile value that the discriminator will be trained to.
      :param n_critic: Interval between epochs to perform a discriminator assessment.
      :param n_critic_stop: The number of critic rounds without improvement after which the training
                            will be stopped.
      :param gen_rounds: The number of generator rounds.
      :param sensitivity_max_col_nums: The maximum number of columns used to compute sensitivity.
      :param use_ks: Whether to use KS test or not.
      :param full_sensitivity: Whether to use full sensitivity or not.
      :param sensitivity_orig_frac_multiple: The size of the training data relative to the chosen
                                             `frac` that will be used in computing the sensitivity. The larger this value is, the
                                             more robust the sensitivity threshold will be. However,
                                             `(sensitivity_orig_frac_multiple + 2)` multiplied by `frac` must be less than 1.
      :param orig_samples_rounds: This is the number of train/hold-out samples that will be used to
                                  compute the epoch sensitivity value.
      :param load_from_best_mean_sensitivity: Whether to load from best mean sensitivity or not.
      :param target_col: The target column name.

      :returns: Trainer


   .. py:method:: _train_with_sensitivity(df: pandas.DataFrame, device: str = 'cuda', num_bootstrap: int = 500, frac: float = 0.165, frac_max_data: int = 10000, qt_max: Union[str, float] = 0.05, qt_max_default: float = 0.05, qt_interval: int = 100, qt_interval_unique: int = 100, distance: sklearn.metrics.pairwise.manhattan_distances = manhattan_distances, quantile: float = 0.95, n_critic: int = 5, n_critic_stop: int = 2, gen_rounds: int = 3, sensitivity_max_col_nums: int = 20, use_ks: bool = False, resume_from_checkpoint: Union[bool, str] = False, full_sensitivity: bool = False, sensitivity_orig_frac_multiple: int = 4, orig_samples_rounds: int = 5, load_from_best_mean_sensitivity: bool = False)


   .. py:method:: _set_up_relational_coder_configs() -> None


   .. py:method:: _fit_relational(out_df: pandas.DataFrame, in_df: pandas.DataFrame, join_on: str, device='cuda')


   .. py:method:: _fit_tabular(df: pandas.DataFrame, device='cuda', num_train_epochs: int = None, target_epochs: int = None) -> transformers.Trainer


   .. py:method:: _build_tabular_trainer(device='cuda', num_train_epochs: int = None, target_epochs: int = None) -> transformers.Trainer


   .. py:method:: sample(n_samples: int = None, input_unique_ids: Optional[Union[pandas.Series, List]] = None, input_df: Optional[pandas.DataFrame] = None, input_ids: Optional[torch.tensor] = None, gen_batch: Optional[int] = 128, device: str = 'cuda', seed_input: Optional[Union[pandas.DataFrame, Dict[str, Any]]] = None, save_samples: Optional[bool] = False, constrain_tokens_gen: Optional[bool] = True, validator: Optional[realtabformer.rtf_validators.ObservationValidator] = None, continuous_empty_limit: int = 10, suppress_tokens: Optional[List[int]] = None, forced_decoder_ids: Optional[List[List[int]]] = None, related_num: Optional[Union[int, List[int]]] = None, **generate_kwargs) -> pandas.DataFrame

      Generate synthetic tabular data samples

      :param n_samples: Number of synthetic samples to generate for the tabular data.
      :param input_unique_ids: The unique identifier that will be used to link the input
                               data to the generated values when sampling for relational data.
      :param input_df: Pandas DataFrame containing the tabular input data.
      :param input_ids: (NOTE: the `input_df` argument is the preferred input)
                        The input_ids that conditions the generation of the relational data.
      :param gen_batch: Controls the batch size of the data generation process. This parameter
                        should be adjusted based on the compute resources.
      :param device: The device used by the generator.
                     Use torch devices, e.g., `cpu`, `cuda`, `mps` (experimental)
      :param seed_input: A dictionary of `col_name:values` for the seed data. Only `col_names`
                         that are actually in the first sequence of the training input will be used.
      :param constrain_tokens_gen: Set whether we impose a constraint at each step of the generation
                                   limited only to valid tokens for the column.
      :param validator: An instance of `ObservationValidator` for validating the generated samples.
                        The validators are applied to observations only, and don't support inter-observation
                        validation. See `ObservationValidator` docs on how to set up a validator.
      :param continuous_invalid_limit: The sampling will raise an exception if
                                       `continuous_empty_limit` empty sample batches have been produced continuously. This
                                       will prevent an infinite loop if the quality of the data generated is not good and
                                       always produces invalid observations.
      :param suppress_tokens: (from docs) A list of tokens that will be supressed at generation.
                              The SupressTokens logit processor will set their log probs to -inf so that they are
                              not sampled. This is a useful feature for imputing missing values.
      :param forced_decoder_ids: (from docs) A list of pairs of integers which indicates a mapping
                                 from generation indices to token indices that will be forced before sampling. For
                                 example, [[1, 123]] means the second generated token will always be a token of
                                 index 123. This is a useful feature for constraining the model to generate only
                                 specific stratification variables in surveys, e.g., GEO1, URBAN/RURAL variables.
      :param related_num: A column name in the input_df containing the number of observations that the child
                          table is expected to have for the parent observation. It can also be an integer if the input_df
                          corresponds to a set of observations having the same number of expected observations.
                          This parameter is only valid for the relational model.
      :param generate_kwargs: Additional keywords arguments that will be supplied to `.generate`
                              method. For a comprehensive list of arguments, see:
                              https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate

      :returns: DataFrame with n_samples rows of generated data


   .. py:method:: predict(data: pandas.DataFrame, target_col: str, target_pos_val: Any = None, batch: int = 32, obs_sample: int = 30, fillunk: bool = True, device: str = 'cuda', disable_progress_bar: bool = True, **generate_kwargs) -> pandas.Series

      Use the trained model to make predictions on a given dataframe.

      :param data: The data to make predictions on, in the form of a Pandas dataframe.
      :param target_col: The name of the target column in the data to predict.
      :param target_pos_val: The positive value in the target column to use for binary
                             classification. This is produces a one-to-many prediction relative to
                             `target_pos_val` for targets that are multi-categorical.
      :param batch: The batch size to use when making predictions.
      :param obs_sample: The number of observations to sample from the data when making predictions.
      :param fillunk: If True, the function will fill any missing values in the data before making
                      predictions. Fill unknown tokens with the mode of the batch in the given step.
      :param device: The device to use for prediction. Can be either "cpu" or "cuda".
      :param \*\*generate_kwargs: Additional keyword arguments to pass to the model's `generate`
                                  method.

      :returns: A Pandas series containing the predicted values for the target column.


   .. py:method:: save(path: Union[str, pathlib.Path], allow_overwrite: Optional[bool] = False)

      Save REaLTabFormer Model

      Saves the model weights and a configuration file in the given directory.
      :param path: Path where to save the model


   .. py:method:: load_from_dir(path: Union[str, pathlib.Path])
      :classmethod:

      Load a saved REaLTabFormer model

      Load trained REaLTabFormer model from directory.
      :param path: Directory where REaLTabFormer model is saved

      :returns: REaLTabFormer instance



