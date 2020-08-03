import csv
import pandas as pd
import json
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filepath', default='', help='path to the original annotation csv file')
parser.add_argument('outfile', default='', help='output file name for the formatted csv')

class XYLatLonConversion:
    def __init__(self, xmin, xmax, ymin, ymax, lonmin, lonmax, latmin, latmax):
        assert xmin != xmax
        assert ymin != ymax
        self.x1 = xmin
        self.x2 = xmax
        self.y1 = ymin
        self.y2 = ymax        

        assert lonmin != lonmax
        assert latmin != latmax
        self.lon1 = lonmin
        self.lon2 = lonmax
        self.lat1 = latmin
        self.lat2 = latmax    

    @staticmethod
    def _linear(x, x1, x2, y1, y2):
        return (y2 * (x - x1) - y1 * (x - x2)) / (x2 - x1)    
    
    def xy_to_latlon(self, x, y):
        return (
            self._linear(y, self.y1, self.y2, self.lat1, self.lat2),
            self._linear(x, self.x1, self.x2, self.lon1, self.lon2),
        )    
    
    def latlon_to_xy(self, lat, lon):
        return (
            self._linear(lon, self.lon1, self.lon2, self.x1, self.x2),
            self._linear(lat, self.lat1, self.lat2, self.y1, self.y2),
        )

cvt = XYLatLonConversion(0,3600,0,1200,0,360,60,-60)


def csv_creator(filepath, outfile):
	annotation_data = pd.read_csv(filepath)

	filenames = list(annotation_data['DateTime'])
	lat_p = list(annotation_data['lat_p'])
	lon_p = list(annotation_data['lon_p'])
	labels = annotation_data['labels']

	fname = []
	xmin = []
	ymin = []
	w = []
	h = []
	supercategory = []
	height = []
	width = []
	category_id = []
	area = []
	crowd = []

	for filename in filenames:
		name = 'ssd_' + filename.split(' ')[0][:4] + filename.split(' ')[0][5:7] + filename.split(' ')[0][8:] + '_' + filename.split(' ')[1][:2] + '.png'
		fname.append(name)

	for x, y in zip(lat_p, lon_p):
		Cx, Cy = cvt.latlon_to_xy(x, y)
		if (Cy - 10) < 0:
			ymin.append(0)
			h.append(int(20 - (-1 * (Cy-10))))
			area.append(int(20 - (-1 * (Cy-10))) * 20)
		else:
			ymin.append(int(Cy - 10))
			h.append(20)
			area.append(400)
		xmin.append(int(Cx - 10))
		w.append(20)
		supercategory.append('none')
		height.append(1201)
		width.append(3600)
		category_id.append(0)
		crowd.append(0)


	fname = pd.Series(fname)
	xmin = pd.Series(xmin)
	ymin = pd.Series(ymin)
	w = pd.Series(w)
	h = pd.Series(h)
	supercategory = pd.Series(supercategory)
	height = pd.Series(height)
	width = pd.Series(width)
	category_id = pd.Series(category_id)
	crowd = pd.Series(crowd)
	area = pd.Series(area)


	counter = collections.Counter(filenames)

	counter_dict = dict(counter)

	temp_dict = {}

	for count, element in enumerate(counter_dict):
		temp_dict[element] = count

	ids = []

	for file in filenames:
		if file in temp_dict.keys():
			ids.append(temp_dict[file])

	ids = pd.Series(ids)

	df = pd.DataFrame()
	df['fname'] = fname
	df['xmin'] = xmin
	df['ymin'] = ymin
	df['w'] = w
	df['h'] = h
	df['name'] = labels
	df['supercategory'] = supercategory
	df['id'] = ids
	df['height'] = height
	df['width'] = width
	df['category_id'] = category_id
	df['iscrowd'] = crowd
	df['area'] = area

	df.to_csv(outfile)


if __name__ == '__main__':
	args = parser.parse_args()
	filepath = args.filepath
	outfile = args.outfile

	csv_creator(filepath, outfile)