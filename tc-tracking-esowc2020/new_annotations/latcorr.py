import pandas as pd
import numpy as np
import copy


file = pd.read_csv('mod2019.csv')

newfile = copy.deepcopy(file)

#delrow = newfile[newfile['ymin'] < 200 or newfile['ymin'] > 1000].index

#newfile = newfile.drop(delrow)


for idx, each in newfile.iterrows():
	if each['ymin'] < 200 or each['ymin'] > 1000:
		newfile = newfile.drop(idx)

# print(file)
# print(newfile)

newfile.to_csv('label19.csv')