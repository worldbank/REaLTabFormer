from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon


class ValidatorBase:
    def __init__(self) -> None:
        pass

    def validate(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError


class RangeValidator(ValidatorBase):
    def __init__(
        self,
        min_val: Union[float, int, np.number],
        max_val: Union[float, int, np.number],
    ) -> None:
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

    def validate(  # type: ignore
        self, val: Union[float, int, np.number], *args: Any, **kwargs: Any
    ) -> bool:

        return self.min_val <= val <= self.max_val


class GeoValidator(ValidatorBase):
    def __init__(self, geo_bound: Union[Polygon, MultiPolygon]) -> None:
        super().__init__()

        self.geo_bound = geo_bound

    def validate(self, lon: float, lat: float) -> bool:  # type: ignore
        p = Point(lon, lat)

        return self.geo_bound.contains(p)


class ObservationValidator(ValidatorBase):
    def __init__(
        self, validators: Optional[Dict[str, Tuple[ValidatorBase, Tuple[str]]]] = None
    ) -> None:
        super().__init__()
        self.validators = validators or {}

    def validate(self, series: pd.Series) -> bool:  # type: ignore
        is_valid = True
        for vname in self.validators:
            validator, cols = self.validators[vname]
            is_valid = is_valid and validator.validate(*(series[c] for c in cols))

            if not is_valid:
                break

        return is_valid

    def validate_df(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.validate, axis=1)

    def add_validator(self, name, validator, cols) -> None:
        self.validators[name] = (validator, cols)

    def remove_validator(self, name) -> Tuple:
        return self.validators.pop(name)
