:py:mod:`realtabformer.rtf_datacollator`
========================================

.. py:module:: realtabformer.rtf_datacollator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.rtf_datacollator.RelationalDataCollator




.. py:class:: RelationalDataCollator

   Data collator that will dynamically pad the inputs received, as well as the labels.
   Adopted from the DataCollatorForSeq2Seq:
    https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/data/data_collator.py#L510

   :param max_length: Maximum length of the returned list and optionally padding length (see above).
   :type max_length: `int`, *optional*
   :param pad_to_multiple_of: If set will pad the sequence to a multiple of the provided value.
                              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                              7.5 (Volta).
   :type pad_to_multiple_of: `int`, *optional*
   :param label_pad_token_id: The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
   :type label_pad_token_id: `int`, *optional*, defaults to -100
   :param return_tensors: The type of Tensor to return. Allowable values are "np", "pt" and "tf".
   :type return_tensors: `str`

   .. py:attribute:: max_length
      :type: Optional[int]

      

   .. py:attribute:: pad_to_multiple_of
      :type: Optional[int]

      

   .. py:attribute:: label_pad_token_id
      :type: int

      

   .. py:attribute:: return_tensors
      :type: str
      :value: 'pt'

      

   .. py:method:: __call__(features, return_tensors=None)



