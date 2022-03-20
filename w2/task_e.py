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
results_dir = Path('./results/task_e/')
results_dir.mkdir(exist_ok=True)

if __name__ == '__main__':

    # FIXME the detection model is NOT a resnet 50 --> Takes super long to infer
    model_list = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', ]
    # 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    kitti_names = [""] * 11
    kitti_names[0] = "background"
    kitti_names[1] = "car"
    kitti_names[2] = "pedestrian"
    kitti_names[10] = "ignore"

    DATASET_NAME = "KITTI-MOTS_"

    for d in ['training', 'val']:
        DatasetCatalog.register(DATASET_NAME + d, lambda d=d: get_KITTI_dataset(dataset_dir, d))
        MetadataCatalog.get(DATASET_NAME + d).set(
            thing_classes=kitti_names, stuff_classes=kitti_names
        )
    metadata = MetadataCatalog.get(DATASET_NAME + "training")

    for model_yaml in model_list:
        print('Creating dataset')
        kitti_meta = MetadataCatalog.get(DATASET_NAME + "val")

        cfg = get_cfg()
        cfg.defrost()
        cfg.merge_from_file(model_zoo.get_config_file(model_yaml))

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATASETS.TRAIN = (DATASET_NAME + "training",)
        cfg.DATASETS.VAL = (DATASET_NAME + "val",)
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 5000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_names)
        cfg.TEST.EVAL_PERIOD = 5
        cfg.OUTPUT_DIR = str(results_dir)

        predictor = DefaultPredictor(cfg)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        print('Evaluating model')

        """ EVALUATION """

        evaluator = COCOEvaluator(DATASET_NAME + "val", output_dir=str(results_dir))
        val_loader = build_detection_test_loader(cfg, DATASET_NAME + "val")

        print(inference_on_dataset(predictor.model, val_loader, evaluator))

        # im = cv2.imread(dataset_dicts[0]["file_name"])
        # outputs = predictor(im)
        # v = Visualizer(im[:, :, ::-1],
        #                metadata=kitti_meta,
        #                scale=0.5,
        #                instance_mode=ColorMode.IMAGE_BW
        # )
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # show_img(out.get_image()[:, :, ::-1])
