import glob
import csv
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', default='', help='Path to csv with annotations')
parser.add_argument('outfile', default='', help='Output file name for JSON annotations')

def refine_json(dict_list):
	"""
	dict_list: images list
	"""

	# Set the same ids for images with same DateTime
	seen_set = set()
	new_list = []
	for list_elem in dict_list:
	    elem_tuple = tuple(sorted(list_elem.items()))
	    if elem_tuple not in seen_set:
	        seen_set.add(elem_tuple)
	        new_list.append(list_elem)
	return new_list


def json_annotator(csv_file, outfile):
	"""
	csv_file: Annotation file
	outfile: output json file name
	"""

	# Read the annotation file
	data_file = pd.read_csv(csv_file)

	# Define the json format
	json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

	# Prepare the json structure
	for idx, row in data_file.iterrows():
		image = {
			"file_name": row['fname'],
			"height": row['height'],
			"width": row['width'],
			"id": row['id']
		}

		json_dict['images'].append(image)

		annot = {
			"id": idx,
			"area": row['area'],
			"iscrowd": row['iscrowd'],
			"image_id": row['id'],
			"bbox": [row['xmin'], row['ymin'], row['w'], row['h']],
			"category_id": int(row['category_id']),
			"ignore": 0,
			"segmentation": [],
		}

		json_dict['annotations'].append(annot)

		#cat = {"supercategory": "none", "id": int(row['name'][-1]), "name": row['name']}
		cat = {"id": int(row['name'][-1]), "name": row['name']}

		json_dict['categories'].append(cat)


	images = json_dict['images']
	cat = json_dict['categories']

	new_img = refine_json(images)
	new_cat = refine_json(cat)

	json_dict = {**json_dict, 'images': new_img}
	json_dict = {**json_dict, 'categories': new_cat}

	json_file = outfile
	json_fp = open(json_file, 'w')
	json_str = json.dumps(json_dict)
	json_fp.write(json_str)
	json_fp.close()

if __name__ == '__main__':
	args = parser.parse_args()
	csv_file = args.csv_file
	outfile = args.outfile

	json_annotator(csv_file, outfile)