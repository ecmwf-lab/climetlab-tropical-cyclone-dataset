# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab import Dataset, load_source
import numpy as np


def normalise_01(a):
    return (a - np.amin(a)) / (np.amax(a) - np.amin(a))


# retrieve, param=clbt, channel=9, date=20160401/to/20181231, ident=57, instrument=207, step=0/6, time=0/12, type=ssd, grid=0.1/0.1
# retrieve, param=tcw


class SimSat(Dataset):

    home_page = "https://github.com/ecmwf/climetlab-ts-dataset"
    documentation = "Work in progress"

    def __init__(self, **req):

        source = load_source("mars", param="2t", levtype="sfc", date="-1")
        self.source = source

        self._fields = []
        for s in source:
            self._fields.append((s, "FIXME_label_from_csv"))

    def fields(self):
        return self._fields

    def title(self, label):
        return "Cloudy brightness temperature"  # "Total column water"

    # load_data is used by keras
    def load_data(self, normalise=True, test_size=0.5, shuffle=False, fields=False):
        data = []
        for field, label in self._fields:
            if normalise:
                array = normalise_01(field.to_numpy())
            else:
                array = field.to_numpy()
            data.append((array, label, field))

        if shuffle:
            raise ValueError("This dataset does not support shuffle.")
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
