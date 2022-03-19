from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from dataset_dict import *
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, cv2, random
from dataset_dict import get_KITTI_dataset

setup_logger()

# FIXME: Add the correct path here
dataset_dir = Path("/home/mcv/datasets/KITTI-MOTS/")
results_dir = Path('./results/task_d/')
os.makedirs(results_dir, exist_ok=True)

if __name__ == '__main__':

    model_list = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                  'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    for d in ['training', 'val']:
        DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_dataset(dataset_dir, d))
        MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["Car", "Pedestrian", "", "", "", "", "", "", "", "", "Ignore"])
    metadata = MetadataCatalog.get("KITTI-MOTS_val")

    for model_yaml in model_list:
        print('Creating dataset')
        dataset_dicts = get_KITTI_dataset(dataset_dir, 'val')

        cfg = get_cfg()
        cfg.defrost()
        cfg.merge_from_file(model_zoo.get_config_file(model_yaml))

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATASETS.VAL = "KITTI-MOTS_val"
        predictor = DefaultPredictor(cfg)

        print('Evaluating model')


        dataset_dicts = get_KITTI_dataset(dataset_dir, 'val')
        i = 1
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            print(d["file_name"])
            print(d)
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imwrite(f"./gt_check{i}", out.get_image()[:, :, ::-1])
            i += 1

        """ EVALUATION """
        evaluator = COCOEvaluator("KITTI-MOTS_val", output_dir=str(results_dir))
        val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
        print(inference_on_dataset(predictor.model, val_loader, evaluator))
