# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab_tc_dataset import TCDataset, local_path


class SimSat(TCDataset):
    def __init__(self, labels=local_path("tc_an.csv"), **req):
        r = dict(
            param="clbt",
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
        r.update(req)

        super().__init__(labels, **r)
