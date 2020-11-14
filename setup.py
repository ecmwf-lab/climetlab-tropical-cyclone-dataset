#!/usr/bin/env python
# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import io
import os

import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


version = "0.0.2"


setuptools.setup(
    name="climetlab-tc-dataset",
    version=version,
    description="CliMetLab Tropical Cyclone datasets plugin",
    long_description=read("README.md"),
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="software.support@ecmwf.int",
    license="Apache License Version 2.0",
    url="https://github.com/ecmwf/climetlab-tc-dataset",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["xarray"],
    zip_safe=True,
    entry_points={
        "climetlab.datasets": [
            "tc-simsat-01  = climetlab_tc_dataset.tc_simsat_01",
            "tc-simsat-025 = climetlab_tc_dataset.tc_simsat_025",
            "tc-simsat-05  = climetlab_tc_dataset.tc_simsat_05",
            "tc-tcw-01     = climetlab_tc_dataset.tc_tcw_01",
            "tc-tcw-025    = climetlab_tc_dataset.tc_tcw_025",
            "tc-tcw-05     = climetlab_tc_dataset.tc_tcw_05",
        ]
    },
    keywords="meteorology",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
    ],
)
