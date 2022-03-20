# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os, cv2
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from dataset_dict import get_KITTI_dataset, get_KITTI_dataset_noignore

# Folders
dataset_dir = Path("/home/pau/Documents/datasets/kitti-mots")
results_dir = Path('./results/task_e/')


def inference(img_path):
    im = cv2.imread(img_path)

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


if __name__ == '__main__':

    model_list = ['COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml']

    kitti_names = [""] * 11
    kitti_names[0] = "background"
    kitti_names[1] = "car"
    kitti_names[2] = "pedestrian"
    kitti_names[10] = "ignore"

    kitti_colors = [(255, 255, 255)] * 11
    kitti_colors[0] = (0, 255, 255)
    kitti_colors[1] = (255, 0, 0)
    kitti_colors[2] = (0, 255, 0)
    kitti_colors[10] = (0, 0, 255)


    DATASET_NAME = "KITTI-MOTS-NOIGNORE_"

    for d in ['training']:
        DatasetCatalog.register(DATASET_NAME + d, lambda d=d: get_KITTI_dataset_noignore(dataset_dir, d))
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=kitti_names,
            stuff_classes=kitti_names,
            thing_colors=kitti_colors,
            stuff_colors=kitti_colors,
        )

    for model_yaml in model_list:

        cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(model_yaml))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_names)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(kitti_names)
        cfg.MODEL.WEIGHTS = str(results_dir / "model_0000999.pth")
        cfg.DATASETS.TRAIN = (DATASET_NAME + "training",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.OUTPUT_DIR = str(results_dir)

        predictor = DefaultPredictor(cfg)

        # model string Detection/InstanceSegmentation
        model_type = model_yaml.split('/')[0].split('-')[-1]

        for seq in ["0014"]:
            out_dir = results_dir / model_yaml.split('/')[0] / seq
            out_dir.mkdir(parents=True, exist_ok=True)
            data_dir = dataset_dir / "training" / "image_02" / seq
            for imfile in data_dir.glob("*"):
                img = inference(str(imfile))
                cv2.imwrite(str(out_dir / imfile.parts[-1]), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
