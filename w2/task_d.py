from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from dataset_dict import *
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from utils import show_img
import os, cv2, random
from dataset_dict import get_KITTI_dataset
from detectron2.structures import Instances
import json

setup_logger()

# FIXME: Add the correct path here
# dataset_dir = Path("/home/mcv/datasets/KITTI-MOTS/")
dataset_dir = Path("/home/pau/Documents/datasets/kitti-mots")
results_dir = Path('./results/task_d/')
os.makedirs(results_dir, exist_ok=True)

if __name__ == '__main__':
    
    # FIXME the detection model is NOT a resnet 50 --> Takes super long to infer
    model_list = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                  'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    # for d in ['training', 'val']:
    #     DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_dataset(dataset_dir, d))
    #     MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["Car", "Pedestrian", "", "", "", "", "", "", "", "", "Ignore"])
    # metadata = MetadataCatalog.get("KITTI-MOTS_val")

    coco_names = [""] * 81
    coco_names[0] = "background"
    coco_names[1] = "person"
    coco_names[3] = "car"

    DATASET_NAME = "KITTI-MOTS-COCO_"
    for d in ['training', 'val']:
        DatasetCatalog.register(DATASET_NAME + d, lambda d=d: get_KITTI_dataset_COCO_ids(dataset_dir, d))
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=coco_names, stuff_classes=coco_names
        )
    metadata = MetadataCatalog.get(DATASET_NAME + "val")

    for model_yaml in model_list:
        print('Creating dataset')
        dataset_dicts = get_KITTI_dataset_COCO_ids(dataset_dir, 'val')

        kitti_meta = MetadataCatalog.get(DATASET_NAME + "val")

        img = cv2.imread(dataset_dicts[0]["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_meta, scale=0.5)
        out = visualizer.draw_dataset_dict(dataset_dicts[0])
        show_img(out.get_image()[:, :, ::-1])

        cfg = get_cfg()
        cfg.defrost()
        cfg.merge_from_file(model_zoo.get_config_file(model_yaml))

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATASETS.VAL = DATASET_NAME + "val"
        predictor = DefaultPredictor(cfg)


        print('Evaluating model')

        """ EVALUATION """

        evaluator = COCOEvaluator(DATASET_NAME + "val", output_dir=str(results_dir))
        val_loader = build_detection_test_loader(cfg, DATASET_NAME + "val")

        print(inference_on_dataset(predictor.model, val_loader, evaluator))
