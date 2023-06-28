:py:mod:`realtabformer.rtf_sampler`
===================================

.. py:module:: realtabformer.rtf_sampler

.. autoapi-nested-parse::

   This module contains the implementation for the sampling
   algorithms used for tabular and relational data generation.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.rtf_sampler.REaLSampler
   realtabformer.rtf_sampler.TabularSampler
   realtabformer.rtf_sampler.RelationalSampler




Attributes
~~~~~~~~~~

.. autoapisummary::

   realtabformer.rtf_sampler.NQ_COL


.. py:data:: NQ_COL
   :value: '_nq_ds_'

   

.. py:class:: REaLSampler(model_type: str, model: transformers.PreTrainedModel, vocab: Dict, processed_columns: List, max_length: int, col_size: int, col_idx_ids: Dict, columns: List, datetime_columns: List, column_dtypes: Dict, column_has_missing: Dict, drop_na_cols: List, col_transform_data: Dict, random_state: Optional[int] = 1029, device='cuda')

   .. py:method:: _prefix_allowed_tokens_fn(batch_id, input_ids) -> List
      :abstractmethod:


   .. py:method:: _convert_to_table(synth_df: pandas.DataFrame) -> pandas.DataFrame


   .. py:method:: _generate(device: torch.device, as_numpy: Optional[bool] = True, constrain_tokens_gen: Optional[bool] = True, **generate_kwargs) -> Union[torch.tensor, numpy.ndarray]


   .. py:method:: _validate_synth_sample(synth_sample: pandas.DataFrame) -> pandas.DataFrame


   .. py:method:: _recover_data_values(synth_sample: pandas.DataFrame) -> pandas.DataFrame


   .. py:method:: _processes_sample(sample_outputs: numpy.ndarray, vocab: Dict, relate_ids: Optional[List[Any]] = None, validator: Optional[realtabformer.rtf_validators.ObservationValidator] = None) -> pandas.DataFrame


   .. py:method:: _validate_data(synth_df: pandas.DataFrame, validator: Optional[realtabformer.rtf_validators.ObservationValidator] = None) -> pandas.DataFrame


   .. py:method:: _validate_missing(synth_df: pandas.DataFrame) -> pandas.DataFrame



.. py:class:: TabularSampler(model_type: str, model: transformers.PreTrainedModel, vocab: Dict, processed_columns: List, max_length: int, col_size: int, col_idx_ids: Dict, columns: List, datetime_columns: List, column_dtypes: Dict, column_has_missing: Dict, drop_na_cols: List, col_transform_data: Dict, random_state: Optional[int] = 1029, device='cuda')

   Bases: :py:obj:`REaLSampler`

   Sampler class for tabular data generation.

   .. py:method:: sampler_from_model(rtf_model, device: str = 'cuda')
      :staticmethod:


   .. py:method:: _prefix_allowed_tokens_fn(batch_id, input_ids) -> List


   .. py:method:: _process_seed_input(seed_input: Union[pandas.DataFrame, Dict[str, Any]]) -> torch.Tensor


   .. py:method:: sample_tabular(n_samples: int, gen_batch: Optional[int] = 128, device: Optional[str] = 'cuda', seed_input: Optional[Union[pandas.DataFrame, Dict[str, Any]]] = None, constrain_tokens_gen: Optional[bool] = True, validator: Optional[realtabformer.rtf_validators.ObservationValidator] = None, continuous_empty_limit: int = 10, suppress_tokens: Optional[List[int]] = None, forced_decoder_ids: Optional[List[List[int]]] = None, **generate_kwargs) -> pandas.DataFrame


   .. py:method:: predict(data: pandas.DataFrame, target_col: str, target_pos_val: Any = None, batch: int = 32, obs_sample: int = 30, fillunk: bool = True, device: str = 'cuda', disable_progress_bar: bool = True, **generate_kwargs) -> pandas.Series

      fillunk: Fill unknown tokens with the mode of the batch.
      target_pos_val: Categorical value for the positive target. This is produces a
       one-to-many prediction relative to `target_pos_val` for targets that are multi-categorical.



.. py:class:: RelationalSampler(model_type: str, model: transformers.PreTrainedModel, vocab: Dict, processed_columns: List, max_length: int, col_size: int, col_idx_ids: Dict, columns: List, datetime_columns: List, column_dtypes: Dict, column_has_missing: Dict, drop_na_cols: List, col_transform_data: Dict, in_col_transform_data: Dict, random_state: Optional[int] = 1029, device='cuda')

   Bases: :py:obj:`REaLSampler`

   Sampler class for relational data generation.

   .. py:method:: sampler_from_model(rtf_model, device: str = 'cuda')
      :staticmethod:


   .. py:method:: sample_relational(input_unique_ids: Union[pandas.Series, List], input_df: Optional[pandas.DataFrame] = None, input_ids: Optional[torch.tensor] = None, gen_batch: Optional[int] = 128, device: Optional[str] = 'cuda', constrain_tokens_gen: Optional[bool] = True, validator: Optional[realtabformer.rtf_validators.ObservationValidator] = None, continuous_empty_limit: Optional[int] = 10, suppress_tokens: Optional[List[int]] = None, forced_decoder_ids: Optional[List[List[int]]] = None, related_num: Optional[Union[int, List[int]]] = None, **generate_kwargs) -> pandas.DataFrame


   .. py:method:: _get_min_max_length(related_num)


   .. py:method:: _sample_input_batch(input_df: Optional[pandas.DataFrame] = None, gen_batch: Optional[int] = 128, device: Optional[str] = 'cuda', constrain_tokens_gen: Optional[bool] = True, suppress_tokens: Optional[List[int]] = None, forced_decoder_ids: Optional[List[List[int]]] = None, **generate_kwargs)


   .. py:method:: _get_relational_col_idx_ids(len_ids: int) -> List

      This method returns the true index given the generation step `i`.

      col_size: The expected number of variables for a single observation.
          This is equal to the number of columns.

      ### Generating constrained tokens per step
      ```
          1 -> BOS
          2 -> BMEM or EOS
          3 -> col 0
          ...
          3 + col_size -> col col_size - 1
          3 + col_size + 1 -> EMEM
          3 + col_size + 2 -> BMEM or EOS
          3 + col_size + 3 -> col 0
      ```


   .. py:method:: _prefix_allowed_tokens_fn(batch_id, input_ids) -> List



