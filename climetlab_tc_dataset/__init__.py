# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from os import path
import numpy as np

from climetlab import Dataset, load_source
from .labels import *
from .coordinates import Coordinates


def normalise_01(a):
    return (a - np.amin(a)) / (np.amax(a) - np.amin(a))


def local_path(filename):
    return path.join(path.dirname(__file__), filename)


def fill_label(coord, l):
    p = l["pressure"]
    l["class"] = 3 if p < 965 else 2 if p < 990 else 1 if p < 1005 else 0
    l["x"], l["y"] = coord.latlon_to_xy(l["lat"], l["lon"])
    return l


class TCDataset(Dataset):

    home_page = "https://github.com/ecmwf/climetlab-ts-dataset"
    documentation = "Work in progress"

    def __init__(self, labels=local_path("tc_an.csv"), **req):

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
        assert len(source) > 0
        self.source = source

        # set coordinate conversion from first field
        with source[0] as g:
            ny, nx = g.shape
            area = req.get("area", [90.0, 0.0, -90.0, 360.0])
            assert len(area) == 4
            self._coord = Coordinates(
                0,
                nx,
                0,
                ny,
                area[1],
                area[3],
                area[0],
                area[2],
            )

        # set labels
        if isinstance(labels, str):
            labels = LabelsFromCSV(labels)
        assert isinstance(labels, Labels), "Unsupported labels '%s' (%s)" % (
            labels,
            type(labels),
        )

        # set fields (labeled)
        self._fields = []
        for s in source:
            l = [fill_label(self._coord, l) for l in labels.lookup(s.valid_datetime())]
            self._fields.append((s, l))

    def fields(self):
        return self._fields

    def title(self, label):
        return self.source[0].metadata().get("shortName", "unknown")

    def grid_definition(self):
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
