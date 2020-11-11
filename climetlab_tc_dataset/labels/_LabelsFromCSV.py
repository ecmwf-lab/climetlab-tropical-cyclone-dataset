# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import pandas as pd
from . import Labels


class LabelsFromCSV(Labels):
    def __init__(self, filename, dt: str = "DateTime"):
        self._dt = dt
        self._df = pd.read_csv(filename)
        self._df[dt] = pd.to_datetime(self._df[dt])

    @property
    def datetime(self):
        return self._df[self._dt]

    def lookup(self, datetime):
        return self._df[self.datetime == datetime].to_dict("records")
