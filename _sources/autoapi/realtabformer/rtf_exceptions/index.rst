:py:mod:`realtabformer.rtf_exceptions`
======================================

.. py:module:: realtabformer.rtf_exceptions


Module Contents
---------------

.. py:exception:: SampleEmptyError(message='Generated sample is empty after validation.', in_size=None)


   Bases: :py:obj:`Exception`

   Exception raised for generated samples without valid observations.

   .. attribute:: salary -- input salary which caused the error

      

   .. attribute:: message -- explanation of the error

      

   .. py:method:: __str__()

      Return str(self).



.. py:exception:: SampleEmptyLimitError(message='Generated sample is still empty after the set limit.', in_size=None)


   Bases: :py:obj:`SampleEmptyError`

   Exception raised when SampleEmptyError is raised
   continuously for some specific limit.


