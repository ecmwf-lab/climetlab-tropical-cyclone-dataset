# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab import Dataset, load_source

from os import path
import numpy as np
import pandas as pd
import datetime


def normalise_01(a):
    return (a - np.amin(a)) / (np.amax(a) - np.amin(a))


class Labels:
    def __init__(self, filename):
        self.df = pd.read_csv(path.join(path.dirname(__file__), filename))

        # keep only: DateTime, lat_p, lon_p, pres
        self.df["DateTime"] = pd.to_datetime(self.df["DateTime"])
        self.df.drop(
            inplace=True, columns=["wind", "lat_w", "lon_w", "step", "windrad", "name"]
        )

    def datetime_min(self):
        return min(self.df["DateTime"])

    def datetime_max(self):
        return max(self.df["DateTime"])

    @staticmethod
    def _class(p):
        if p < 965:
            return 3
        if p < 990:
            return 2
        if p < 1005:
            return 1
        return 0

    def label(self, date):
        l = []
        for row in self.df[self.df["DateTime"] == date].itertuples():
            l.append(
                {"lat": row.lat_p, "lon": row.lon_p, "class": self._class(row.pres)}
            )
        return l


# labels file
labels = Labels("tc_an.csv")
date = "{}/to/{}".format(labels.datetime_min().date(), labels.datetime_max().date())
print("labels: date={}".format(date))


class SimSat(Dataset):

    home_page = "https://github.com/ecmwf/climetlab-ts-dataset"
    documentation = "Work in progress"

    def __init__(self, **req):

        # retrieve, param=tcw
        source = load_source(
            "mars",
            param="clbt",
            date=date,
            time=[0, 12],
            step=[0, 6],
            grid=[0.1, 0.1],
            area=[60.0, 0.0, -60.0, 360.0],
            type="ssd",
            channel="9",
            ident="57",
            instrument="207",
        )
        self.source = source

        self._fields = []
        for c, s in enumerate(source):
            d = s.datetime() + datetime.timedelta(hours=[0, 6][c % 2])
            self._fields.append((s, labels.label(d)))

    def fields(self):
        return self._fields

    def title(self, label):
        # return "Total column water"
        return "Cloudy brightness temperature"

    # load_data is used by keras
    def load_data(self, normalise=True, test_size=0.74, shuffle=False, fields=False):
        data = []
        for field, label in self._fields:
            if normalise:
                array = normalise_01(field.to_numpy())
            else:
                array = field.to_numpy()
            data.append((array, label, field))

        if shuffle:
            raise ValueError("This dataset does not support shuffle (intentionally).")
        half = int(len(data) * (1.0 - test_size))

        x_train, y_train, f_train = (
            np.array([x[0] for x in data[:half]]),
            np.array([x[1] for x in data[:half]]),
            [x[2] for x in data[:half]],
        )

        x_test, y_test, f_test = (
            np.array([x[0] for x in data[half:]]),
            np.array([x[1] for x in data[half:]]),
            [x[2] for x in data[half:]],
        )

        if fields:
            return (x_train, y_train, f_train), (x_test, y_test, f_test)

        return (x_train, y_train), (x_test, y_test)


dataset = SimSat
