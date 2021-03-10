# Annotation formatter file

import collections
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('filepath', default='', help='path to the original annotation csv file') #newlabels.csv
parser.add_argument('outfile', default='', help='output file name for the formatted csv')
parser.add_argument('lon', default='', help='longitude value')
parser.add_argument('lat', default='', help='latitude value')
parser.add_argument('bbox_size', default='', help='bounding box size')
#parser.add_argument('--input_size', default='', help='Provide the input image size as a tuple (lon, lat)')

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

#cvt = XYLatLonConversion(0,3600,0,1200,0,360,60,-60)
#cvt = XYLatLonConversion(0,1440,0,481,0,360,60,-60)

def name_creator(filenames_list):
    """
    filenames_list: list of filenames
    """

    fname = []

    # Prepare the names in the required format
    for filename in filenames_list:
        name = 'ssd_' + filename.split(' ')[0][:4] + filename.split(' ')[0][5:7] + filename.split(' ')[0][8:] + '_' + filename.split(' ')[1][:2] + '.png'
        fname.append(name)
    return fname

def csv_creator(filepath, outfile, lon, lat, bbox_size):
    """
    filepath: Original annotation csv file
    outfile: Revised csv file name
    lon: Longitude max value in input
    lat: Latitude max value in input
    """

    # Read the original CSV
    
    annotation_data = pd.read_csv(filepath)

    # Get the required data series from dataframe as lists

    filenames = list(annotation_data['DateTime'])
    lat_p = list(annotation_data['lat_p'])
    lon_p = list(annotation_data['lon_p'])
    labels = annotation_data['labels']

    # Declare all the new lists to be populated
    fname = []
    xmin = []
    ymin = []
    w = []
    h = []
    height = []
    width = []
    area = []
    crowd = []

    # Name convention for the files
    fname = name_creator(filenames)

    # Handle cases where lon values exceed the limits
    for x, y in zip(lat_p, lon_p):
        Cx, Cy = cvt.latlon_to_xy(x, y)
        if (Cy - (bbox_size/2)) < 0:
            ymin.append(0)
            h.append(int(bbox_size - (-1 * (Cy-(bbox_size/2)))))
            area.append(int(bbox_size - (-1 * (Cy-(bbox_size/2)))) * bbox_size)
        else:
            ymin.append(int(Cy - (bbox_size/2)))
            h.append(bbox_size)
            area.append(bbox_size * bbox_size)
        xmin.append(int(Cx - (bbox_size/2)))
        w.append(bbox_size)
        height.append(lat)
        width.append(lon)
        crowd.append(0)

    # Convert all the lists to series
    fname = pd.Series(fname)
    xmin = pd.Series(xmin)
    ymin = pd.Series(ymin)
    w = pd.Series(w)
    h = pd.Series(h)
    height = pd.Series(height)
    width = pd.Series(width)
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
    df['id'] = ids
    df['height'] = height
    df['width'] = width
    df['iscrowd'] = crowd
    df['area'] = area

    df.to_csv(outfile)

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = args.filepath
    outfile = args.outfile

    #input_size = args.input_size
    lon = int(args.lon)
    lat = int(args.lat)
    bbox_size = int(args.bbox_size)

    #lon, lat = int(input_size[0]), int(input_size[1])

    cvt = XYLatLonConversion(0, lon, 0, lat, 0, 360, 60, -60)

    csv_creator(filepath, outfile, lon, lat, bbox_size)
