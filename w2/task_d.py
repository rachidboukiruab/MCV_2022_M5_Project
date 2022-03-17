from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from dataset_dict import *
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, cv2, random
from task_c import inference

setup_logger()


dataset_dir = '/home/mcv/datasets/KITTI-MOTS/'
results_dir = './results/task_d/'
os.makedirs(results_dir, exist_ok=True)



if __name__ == '__main__':

    model_list = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    for d in ["training", "val"]:
        DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_dataset(dataset_dir, d))
        MetadataCatalog.get("KITTI-MOT_" + d).set(thing_classes=['Car','Pedestrian'])
    kitti_mots_metadata = MetadataCatalog.get('KITTI-MOTS_train')
    
    for model_yalm in model_list:

        dataset_dicts = get_KITTI_dataset(dataset_dir, 'val')

        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file(model_yalm))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yalm)
        predictor = DefaultPredictor(cfg)

        confidence_threshold = 0.5

        """ INFERENCE """

        for d in random.sample(dataset_dicts, 3):    
            im = d["file_name"]
            img = inference(im)

            out_path = results_dir+'/'+model_yalm
            os.makedirs(out_path+'/'+im, exist_ok=True)
        
            cv2.imwrite(out_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            print(f"Processed img {im} for {model_yalm}")

        """ EVALUATION """
        evaluator = COCOEvaluator("KITTI-MOTS_val", output_dir=results_dir)
        val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
        print(inference_on_dataset(predictor.model, val_loader, evaluator))


