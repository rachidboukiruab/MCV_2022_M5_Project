from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import os, cv2, random
from typing_extensions import TypedDict
import json



""" FEATURE  INFERENCE using MASK R-CNN
    reference: The Elephant in the room https://arxiv.org/pdf/1808.03305.pdf fig.3."""


# Folders
dataset_dir = '/home/group01/MCV_2022_M5_Project/w3/task_d/images/'
results_dir = './results/'
os.makedirs(results_dir, exist_ok=True)


def inference(img_path):
    im = cv2.imread(img_path)

    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    return out.get_image()[:, :, ::-1]

class DatasetSplit(TypedDict):
    """
    A typed dict to represent experiment settings. Types should match those in
    the configuration JSON file used as input parameter.
    """
    gt_img: str
    background_1: str
    black: str
    totally_black:str
    noise: str
    background2:str
    background3:str
    background4: str


if __name__ == '__main__':

    with open('img_paths.json') as jsonfile:
        paths = json.load(jsonfile)

    model_yalm = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

    cfg = get_cfg() 

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_yalm))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yalm)
    predictor = DefaultPredictor(cfg)

    # Run inference with pre-trained Mask R-CNN
    for file in paths:
        print(f"Feature inference to image: {file}")
        if paths[file] == 'None':
            break
        else:
            img_path = os.path.join(dataset_dir,paths[file])
            out_path2 = os.path.join(results_dir, paths[file])

            #inference image 
            img = inference(img_path)

            cv2.imwrite(out_path2, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            



    




