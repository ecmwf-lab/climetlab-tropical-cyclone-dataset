# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab import Dataset, load_source, load_dataset

from os import path
import numpy as np
import pandas as pd


def normalise_01(a):
    return (a - np.amin(a)) / (np.amax(a) - np.amin(a))


class Coordinates:
    def __init__(self, x1, x2, y1, y2, lon1, lon2, lat1, lat2):
        assert x1 != x2 and y1 != y2
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)

        assert lon1 != lon2 and lat1 != lat2
        self.lon1 = float(lon1)
        self.lon2 = float(lon2)
        self.lat1 = float(lat1)
        self.lat2 = float(lat2)

    @staticmethod
    def _linear(x, x1, x2, y1, y2):
        return (y2 * (x - x1) - y1 * (x - x2)) / (x2 - x1)

    @staticmethod
    def _delta_linear(x, x1, x2, y1, y2):
        return abs(x * (y2 - y1) / (x2 - x1))

    def xy_to_latlon(self, x, y):
        return (
            self._linear(y, self.y1, self.y2, self.lat1, self.lat2),
            self._linear(x, self.x1, self.x2, self.lon1, self.lon2),
        )

    def latlon_to_xy(self, lat, lon):
        return (
            int(round(self._linear(lon, self.lon1, self.lon2, self.x1, self.x2))),
            int(round(self._linear(lat, self.lat1, self.lat2, self.y1, self.y2))),
        )

    def dxy_to_dlatlon(self, dx, dy):
        return (
            self._delta_linear(dy, self.y1, self.y2, self.lat1, self.lat2),
            self._delta_linear(dx, self.x1, self.x2, self.lon1, self.lon2),
        )

    def dlatlon_to_dxy(self, dlat, dlon):
        return (
            int(
                round(self._delta_linear(dlon, self.lon1, self.lon2, self.x1, self.x2))
            ),
            int(
                round(self._delta_linear(dlat, self.lat1, self.lat2, self.y1, self.y2))
            ),
        )


class Labels:
    def __init__(self, filename, dt: str = "DateTime"):
        self._dt = dt
        self._df = pd.read_csv(filename)
        self._df[dt] = pd.to_datetime(self._df[dt])

    @property
    def datetime(self):
        return self._df[self._dt]

    def lookup(self, datetime):
        return self._df[self.datetime == datetime].to_dict("records")


def klass(p):
    if p < 965:
        return 3
    if p < 990:
        return 2
    if p < 1005:
        return 1
    return 0


class SimSat(Dataset):

    home_page = "https://github.com/ecmwf/climetlab-ts-dataset"
    documentation = "Work in progress"

    def __init__(self, **req):

        # set source(s)
        source = load_source(
            "mars",
            param="clbt",  # tcw
            date="2016-04-01/to/2019-12-31",
            time=[0, 12],
            step=[0, 6],
            grid=[0.1, 0.1],
            area=[60.0, 0.0, -60.0, 360.0],
            type="ssd",
            channel="9",
            ident="57",
            instrument="207",
        )
        # source = load_source(
        #     "mars",
        #     param="t",
        #     level=1000,
        #     date="-1",
        #     type="fc",
        #     time=[0, 12],
        #     step=[0, 6],
        #     grid=[3.0, 3.0],
        # )
        self.source = source

        # set coordinate conversion from first field
        assert len(source) > 0
        with source[0] as g:
            grid = g.grid_definition()
            ny, nx = g.shape
            self._coord = Coordinates(
                0,
                nx,
                0,
                ny,
                grid["west"],
                grid["east"],
                grid["north"],
                grid["south"],
            )

        # set fields and labels
        self._labels = Labels(path.join(path.dirname(__file__), "tc_an.csv"))
        # print("labels: date={}/to/{}".format(min(self._labels.datetime).date(), max(self._labels.datetime).date()))

        self._fields = []
        for s in source:
            labels = self._labels.lookup(s.valid_datetime())
            for l in labels:
                l["x"], l["y"] = self._coord.latlon_to_xy(l["lat_p"], l["lon_p"])
                l["class"] = klass(l["pres"])
            self._fields.append((s, labels))

    def fields(self):
        return self._fields

    def title(self, label):
        # return "Total column water"
        return "Cloudy brightness temperature"

    def grid_definition(self):
        assert len(self.source) > 0
        return self.source[0].grid_definition()

    # load_data is used by keras
    def load_data(self, normalise=True, test_size=0.2664, fields=False):
        data = []
        for field, label in self._fields:
            if normalise:
                array = normalise_01(field.to_numpy())
            else:
                array = field.to_numpy()
            data.append((array, label, field))

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


if __name__ == "__main__":
    print("Hi!")
    ds = load_dataset("tc-dataset")
    print("Bye!")
    # cvt = Coordinates(0, 3600, 0, 1200, 0, 360, 60, -60)
    # cvt = Coordinates(0, 1440, 0, 481, 0, 360, 60, -60)
