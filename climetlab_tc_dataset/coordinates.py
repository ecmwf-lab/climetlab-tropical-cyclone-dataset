# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


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
