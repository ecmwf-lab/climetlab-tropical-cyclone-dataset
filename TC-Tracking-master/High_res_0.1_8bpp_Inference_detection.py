#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive

drive.mount("/content/drive")


# In[ ]:


#!pip install pyyaml==5.1 pycocotools>=2.0.1
import torch, torchvision

print(torch.__version__, torch.cuda.is_available())
get_ipython().system("gcc --version")


# In[ ]:


assert torch.__version__.startswith("1.6")
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html


# In[ ]:


import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[ ]:


get_ipython().run_line_magic("cd", "drive/My Drive/cyc3600")


# In[ ]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml")
)
# cfg.merge_from_file(
#     "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# )
# cfg.DATASETS.TRAIN = ("cyclone_ds_train",)
# cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = (
    2000  # 300 iterations seems good enough, but you can certainly train longer
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    1280  # faster, and good enough for this toy dataset
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 3 classes (data, fig, hazelnut)
cfg.INPUT.MAX_SIZE_TEST = 3600
cfg.INPUT.MAX_SIZE_TRAIN = 3600
cfg.INPUT.MIN_SIZE_TEST = 1201
cfg.INPUT.MIN_SIZE_TRAIN = 1201
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[60, 70, 80, 90]]


# In[ ]:


from detectron2.modeling import build_model

model = build_model(cfg)


# In[ ]:


from detectron2.checkpoint import DetectionCheckpointer

# DetectionCheckpointer(model).load('Mod-DC5-R101-b80/model_final.pth')
DetectionCheckpointer(model).load("b80-R101-DC5-3x/model_final.pth")


# In[ ]:


import torch
from PIL import Image

pred = DefaultPredictor(cfg)


# image = Image.open('test/ssd_20201006_00.png')
# #X = torch.from_numpy(np.array(image))

# X = torch.Tensor(np.array(image))

# print(type(X))
# #exit()

# inputs = [{"image": X}]

# outputs = model(inputs)


# In[ ]:


import glob

flist = glob.glob("test/*")
for each in flist:
    print(each)
    input = cv2.imread(each)
    out = pred(input)
    print(out)


# In[ ]:
