"""annotation_handler"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_file", default="", help="Path to the annotation_file")
parser.add_argument("--data_dir", default="", help="Path to data dir")

# annotation_file = pd.read_csv('~/esowc/satimg/tc_an_2016.csv')


def get_annotation_csv(annotation_file, data_dir):
    """Get image pixel coordinates
    based on lat, lon values."""

    annotation_file = pd.read_csv(annotation_file)
    lon_center_list = annotation_file["lon_p"]
    lat_center_list = annotation_file["lat_p"]

    # Labels
    cyc_name_list = annotation_file["name"]

    filname_list = annotation_file["DateTime"]

    # 2016-04-01 06:00:00
    # ssd_20160401_06
    path_list = []

    for each in filname_list:
        path_list_name = each.split(" ")[0].split("-")
        time = each.split(" ")[1].split(":")[0]
        path_list_name = (
            "ssd_"
            + str(path_list_name[0])
            + str(path_list_name[1])
            + str(path_list_name[2])
            + "_"
            + str(time)
        )
        path_list_name = data_dir + path_list_name
        path_list.append(path_list_name)

    path_series = pd.Series(path_list)

    coord_list_xmin = []
    coord_list_xmax = []
    coord_list_ymin = []
    coord_list_ymax = []

    # Calculations for image size 3600*1201

    for centerX, centerY in zip(lon_center_list, lat_center_list):
        xmin = 10 * (centerX) - 10
        xmax = 10 * (centerX) + 10

        if centerY >= 0:
            ymin = 10 * (centerY) - 10
            ymax = 10 * (centerY) + 10
        else:
            ymin = 10 * (abs(centerY) + 60) - 10
            ymax = 10 * (abs(centerY) + 60) + 10

        coord_list_xmin.append(xmin)
        coord_list_xmax.append(xmax)
        coord_list_ymin.append(ymin)
        coord_list_ymax.append(ymax)

    xmin_series = pd.Series(coord_list_xmin)
    xmax_series = pd.Series(coord_list_xmax)
    ymin_series = pd.Series(coord_list_ymin)
    ymax_series = pd.Series(coord_list_ymax)

    labels_list = annotation_file["name"]
    labels_series = pd.Series(labels_list)

    df = pd.DataFrame(
        {
            "path": path_series,
            "xmin": xmin_series,
            "ymin": ymin_series,
            "xmax": xmax_series,
            "ymax": ymax_series,
            "labels": labels_series,
        }
    )

    df.to_csv("annot_format.csv")
    # for centerY in lat_center_list:
    # 	if centerY >= 0:
    # 		ymin = 10*(centerY) - 10
    # 		ymax = 10*(centerY) + 10
    # 	else:
    # 		ymin = 10 * (abs(centerY)+60) - 10
    # 		ymax = 10 * (abs(centerY)+60) + 10


if __name__ == "__main__":
    args = parser.parse_args()
    annotation_file = args.annotation_file
    data_dir = args.data_dir

    get_annotation_csv(annotation_file, data_dir)
