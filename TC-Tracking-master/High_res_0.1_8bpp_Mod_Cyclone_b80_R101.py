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


from detectron2.data.datasets import register_coco_instances

register_coco_instances("cyclone_ds_train", {}, "modtrain.json", "train")
# register_coco_instances('temp_train', {}, 'ftrain80.json', 'train')


# In[ ]:


cyclone_metadata = MetadataCatalog.get("cyclone_ds_train")
# cyclone_metadata = MetadataCatalog.get('temp_train')


# In[ ]:


dataset_dicts = DatasetCatalog.get("cyclone_ds_train")


# In[ ]:


# import random
# from detectron2.utils.visualizer import Visualizer

# for d in dataset_dicts:
#     if d['file_name'] == 'train/ssd_20180604_12.png':
#         #print(d['file_name'])
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=cyclone_metadata, scale=0.5)
#         vis = visualizer.draw_dataset_dict(d)
#         cv2_imshow(vis.get_image()[:, :, ::-1])


# In[ ]:


import random
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 6):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cyclone_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])


# In[ ]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"))
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)
# cfg.merge_from_file(
#     "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# )
cfg.DATASETS.TRAIN = ("cyclone_ds_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
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


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"))
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)
# cfg.merge_from_file(
#     "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# )
cfg.DATASETS.TRAIN = ("cyclone_ds_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
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


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# In[ ]:


register_coco_instances("cyclone_ds_val", {}, "modval.json", "val")


# In[ ]:


val_metadata = MetadataCatalog.get("cyclone_ds_val")
val_dicts = DatasetCatalog.get("cyclone_ds_val")


# In[ ]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = os.path.join('b80-R101-DC5-3x', "model_final.pth")
# cfg.MODEL.WEIGHTS = '/content/drive/My Drive/cyc1440/output/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set the testing threshold for this model
cfg.DATASETS.TEST = ("cyclone_ds_val",)
predictor = DefaultPredictor(cfg)


# In[ ]:


from detectron2.utils.visualizer import ColorMode

for d in val_dicts:
    print(d["file_name"])
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)

# for d in random.sample(val_dicts, 15):
#     print(d['file_name'])
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     print(outputs)


# In[ ]:


from detectron2.utils.visualizer import ColorMode

for d in random.sample(val_dicts, 30):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)


# In[ ]:


# from detectron2.modeling import build_model
# model = build_model(cfg)

# from detectron2.checkpoint import DetectionCheckpointer
# DetectionCheckpointer(model).load('b80-R101-DC5-3x/model_final.pth')

temp_val_dict = val_dicts[:2]

print(temp_val_dict)
for each in temp_val_dict:
    print(each)

# model.eval()
# with torch.no_grad():
#     outputs = model(val_dicts[:10])

# for d in random.sample(val_dicts, 10):
#     print(d)
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     print(outputs)


# In[ ]:


for d in random.sample(dataset_dicts, 6):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cyclone_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])


from detectron2.utils.visualizer import ColorMode

dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
