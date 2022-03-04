#!/usr/bin/env python3

import numpy as np
import pandas as pd
import metview as mv

start_date = pd.to_datetime("20201008")
end_date = pd.to_datetime("20201008")
byday = 1
datelist_all = pd.date_range(start_date, end_date, freq=str(byday) + "D").tolist()

outpath = "/var/tmp/ne5/sowc2020/"
for date1 in datelist_all:
    data = mv.retrieve(
        channel=9,
        Class="od",
        date=date1,
        expver=1,
        ident=57,
        instrument=207,
        param="260510",
        step=[0, 6],
        stream="oper",
        time=[0, 12],
        type="ssd",
        grid=[0.1, 0.1],
    )

    for data1 in data:
        dateS = mv.valid_date(data1).strftime("%Y%m%d")
        timeS = mv.valid_date(data1).strftime("%H")
        dstring = dateS + "_" + timeS
        print(dstring)
        mv.write(outpath + "ssd_" + dstring + ".grb", data1)


