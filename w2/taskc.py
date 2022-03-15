# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Folders
dataset_dir = '/home/mcv/datasets/KITTI-MOTS/'
results_dir = '../results/task_b/'
'''if not os.path.exists(results_dir):
        os.mkdir(results_dir)'''

if __name__ == '__main__':
    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    

    # Run inference with pre-trained Faster R-CNN (detection) and Mask R-CNN(detection and segmentation) on KITTI-MOTS dataset
    for (root,dirs,files) in os.walk(dataset_dir, topdown=True):
        print (root)
        print (dirs)
        print (files)
        print ('--------------------------------')
        '''results_path = os.path.join(results_dir, root)
        os.makedirs(results_path, exist_ok=True)
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(results_path, file)

            im = cv2.imread(input_path)

            outputs = predictor(im)

            # look at the outputs.
            # print(outputs["instances"].pred_classes)
            # print(outputs["instances"].pred_boxes)

            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])'''