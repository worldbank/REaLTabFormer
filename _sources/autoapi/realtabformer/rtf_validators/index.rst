:py:mod:`realtabformer.rtf_validators`
======================================

.. py:module:: realtabformer.rtf_validators


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.rtf_validators.ValidatorBase
   realtabformer.rtf_validators.RangeValidator
   realtabformer.rtf_validators.GeoValidator
   realtabformer.rtf_validators.ObservationValidator




.. py:class:: ValidatorBase


   .. py:method:: validate(*args: Any, **kwargs: Any) -> bool
      :abstractmethod:



.. py:class:: RangeValidator(min_val: Union[float, int, numpy.number], max_val: Union[float, int, numpy.number])


   Bases: :py:obj:`ValidatorBase`

   .. py:method:: validate(val: Union[float, int, numpy.number], *args: Any, **kwargs: Any) -> bool



.. py:class:: GeoValidator(geo_bound: Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon])


   Bases: :py:obj:`ValidatorBase`

   .. py:method:: validate(lon: float, lat: float) -> bool



.. py:class:: ObservationValidator(validators: Optional[Dict[str, Tuple[ValidatorBase, Tuple[str]]]] = None)


   Bases: :py:obj:`ValidatorBase`

   .. py:method:: validate(series: pandas.Series) -> bool


   .. py:method:: validate_df(df: pandas.DataFrame) -> pandas.Series


   .. py:method:: add_validator(name, validator, cols) -> None


   .. py:method:: remove_validator(name) -> Tuple



