:py:mod:`realtabformer.data_utils`
==================================

.. py:module:: realtabformer.data_utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.data_utils.TabularArtefact
   realtabformer.data_utils.ModelFileName
   realtabformer.data_utils.ModelType
   realtabformer.data_utils.ColDataType
   realtabformer.data_utils.SpecialTokens



Functions
~~~~~~~~~

.. autoapisummary::

   realtabformer.data_utils.get_uuid
   realtabformer.data_utils.fix_multi_decimal
   realtabformer.data_utils.build_vocab
   realtabformer.data_utils.process_numeric_data
   realtabformer.data_utils.process_datetime_data
   realtabformer.data_utils.process_categorical_data
   realtabformer.data_utils.encode_partition_numeric_col
   realtabformer.data_utils.decode_partition_numeric_col
   realtabformer.data_utils.tokenize_numeric_col
   realtabformer.data_utils.encode_column_values
   realtabformer.data_utils.decode_column_values
   realtabformer.data_utils.encode_processed_column
   realtabformer.data_utils.decode_processed_column
   realtabformer.data_utils.extract_processed_column
   realtabformer.data_utils.is_numeric_col
   realtabformer.data_utils.is_datetime_col
   realtabformer.data_utils.is_categorical_col
   realtabformer.data_utils.is_numeric_datetime_col
   realtabformer.data_utils.process_data
   realtabformer.data_utils.get_token_id
   realtabformer.data_utils.get_input_ids
   realtabformer.data_utils.make_dataset
   realtabformer.data_utils.get_relational_input_ids
   realtabformer.data_utils.make_relational_dataset



Attributes
~~~~~~~~~~

.. autoapisummary::

   realtabformer.data_utils.TEACHER_FORCING_PRE
   realtabformer.data_utils.SPECIAL_COL_SEP
   realtabformer.data_utils.NUMERIC_NA_TOKEN
   realtabformer.data_utils.INVALID_NUMS_RE


.. py:data:: TEACHER_FORCING_PRE
   :value: '_TEACHERFORCING'

   

.. py:data:: SPECIAL_COL_SEP
   :value: '___'

   

.. py:data:: NUMERIC_NA_TOKEN
   :value: '@'

   

.. py:data:: INVALID_NUMS_RE
   :value: '[^\\-.0-9]'

   

.. py:class:: TabularArtefact


   .. py:attribute:: best_disc_model
      :type: str
      :value: 'best-disc-model'

      

   .. py:attribute:: mean_best_disc_model
      :type: str
      :value: 'mean-best-disc-model'

      

   .. py:attribute:: not_best_disc_model
      :type: str
      :value: 'not-best-disc-model'

      

   .. py:attribute:: last_epoch_model
      :type: str
      :value: 'last-epoch-model'

      

   .. py:method:: artefacts()
      :staticmethod:



.. py:class:: ModelFileName


   .. py:attribute:: rtf_config_json
      :type: str
      :value: 'rtf_config.json'

      

   .. py:attribute:: rtf_model_pt
      :type: str
      :value: 'rtf_model.pt'

      

   .. py:method:: names()
      :staticmethod:



.. py:class:: ModelType


   .. py:attribute:: tabular
      :type: str
      :value: 'tabular'

      

   .. py:attribute:: relational
      :type: str
      :value: 'relational'

      

   .. py:method:: types()
      :staticmethod:



.. py:class:: ColDataType


   .. py:attribute:: NUMERIC
      :type: str
      :value: 'NUMERIC'

      

   .. py:attribute:: DATETIME
      :type: str
      :value: 'DATETIME'

      

   .. py:attribute:: CATEGORICAL
      :type: str
      :value: 'CATEGORICAL'

      

   .. py:method:: types()
      :staticmethod:



.. py:class:: SpecialTokens


   .. py:attribute:: UNK
      :type: str
      :value: '[UNK]'

      

   .. py:attribute:: SEP
      :type: str
      :value: '[SEP]'

      

   .. py:attribute:: PAD
      :type: str
      :value: '[PAD]'

      

   .. py:attribute:: CLS
      :type: str
      :value: '[CLS]'

      

   .. py:attribute:: MASK
      :type: str
      :value: '[MASK]'

      

   .. py:attribute:: BOS
      :type: str
      :value: '[BOS]'

      

   .. py:attribute:: EOS
      :type: str
      :value: '[EOS]'

      

   .. py:attribute:: BMEM
      :type: str
      :value: '[BMEM]'

      

   .. py:attribute:: EMEM
      :type: str
      :value: '[EMEM]'

      

   .. py:attribute:: RMASK
      :type: str
      :value: '[RMASK]'

      

   .. py:attribute:: SPTYPE
      :type: str
      :value: '[SPTYPE]'

      

   .. py:method:: tokens()
      :staticmethod:



.. py:function:: get_uuid()


.. py:function:: fix_multi_decimal(v)


.. py:function:: build_vocab(df: pandas.DataFrame = None, special_tokens=None, add_columns: bool = True)


.. py:function:: process_numeric_data(series: pandas.Series, max_len: int = 10, numeric_precision: int = 4, transform_data: Dict = None) -> Tuple[pandas.Series, Dict]


.. py:function:: process_datetime_data(series, transform_data: Dict = None) -> Tuple[pandas.Series, Dict]


.. py:function:: process_categorical_data(series: pandas.Series) -> pandas.Series


.. py:function:: encode_partition_numeric_col(col, tr, col_zfill)


.. py:function:: decode_partition_numeric_col(partition_col)


.. py:function:: tokenize_numeric_col(series: pandas.Series, nparts=2, col_zfill=2)


.. py:function:: encode_column_values(series)


.. py:function:: decode_column_values(series)


.. py:function:: encode_processed_column(idx, dtype, col)


.. py:function:: decode_processed_column(col)


.. py:function:: extract_processed_column(col)


.. py:function:: is_numeric_col(col)


.. py:function:: is_datetime_col(col)


.. py:function:: is_categorical_col(col)


.. py:function:: is_numeric_datetime_col(col)


.. py:function:: process_data(df: pandas.DataFrame, numeric_max_len=10, numeric_precision=4, numeric_nparts=2, first_col_type=None, col_transform_data: Dict = None, target_col: str = None) -> Tuple[pandas.DataFrame, Dict]


.. py:function:: get_token_id(token: str, vocab_token2id: Dict[str, int], mask_rate: float = 0) -> int


.. py:function:: get_input_ids(example, vocab: Dict, columns: List, mask_rate: float = 0, return_label_ids: Optional[bool] = True, return_token_type_ids: Optional[bool] = False, affix_bos: Optional[bool] = True, affix_eos: Optional[bool] = True) -> Dict


.. py:function:: make_dataset(df: pandas.DataFrame, vocab: Dict, mask_rate: float = 0, affix_eos: bool = True, return_token_type_ids: bool = False) -> datasets.Dataset


.. py:function:: get_relational_input_ids(example, input_idx, vocab, columns, output_dataset, in_out_idx, output_max_length: Optional[int] = None, return_token_type_ids: bool = False) -> dict


.. py:function:: make_relational_dataset(in_df: pandas.DataFrame, out_df: pandas.DataFrame, vocab: dict, in_out_idx: dict, mask_rate=0, output_max_length: Optional[int] = None, return_token_type_ids: bool = False) -> datasets.Dataset


