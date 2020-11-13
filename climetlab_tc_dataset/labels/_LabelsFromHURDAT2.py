# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import pandas as pd
import datetime

from . import Labels
from climetlab import load_dataset
from climetlab.utils import datetime as dt


class LabelsFromHURDAT2(Labels):
    def __init__(self, dt: str = "time", basins=["atlantic", "pacific"]):
        self._dt = dt
        self._df = pd.DataFrame()

        for basin in basins:
            if self._df.empty:
                self._df = load_dataset("hurricane-database", basin).to_pandas()
            else:
                self._df = pd.concat(
                    [self._df, load_dataset("hurricane-database", basin).to_pandas()],
                    ignore_index=True,
                )

        assert not self._df.empty
        self._df.sort_values(by=dt)

    @property
    def datetime(self):
        return self._df[self._dt]

    def lookup(self, datetime):
        return self._df[self.datetime == dt.to_datetime(datetime)].to_dict("records")
