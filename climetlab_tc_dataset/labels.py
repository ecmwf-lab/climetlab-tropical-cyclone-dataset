# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import pandas as pd

from climetlab import load_dataset


class Labels:
    @property
    def datetime(self):
        return self._df[self._dt]

    def lookup(self, datetime):
        return self._df[self.datetime == datetime].to_dict("records")


class LabelsFromCSV(Labels):
    def __init__(self, filename, dt="DateTime", p="pres", lat="lat_p", lon="lon_p"):
        self._dt = dt
        self._df = pd.read_csv(filename)

        self._df[dt] = pd.to_datetime(self._df[dt])
        self._df.rename(columns={p: "pressure", lat: "lat", lon: "lon"}, inplace=True)


class LabelsFromHURDAT2(Labels):
    def __init__(self, dt="time", basins=["atlantic", "pacific"]):
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

        assert all([c in self._df.columns for c in [dt, "lat", "lon", "pressure"]])
        self._df.sort_values(by=dt)
