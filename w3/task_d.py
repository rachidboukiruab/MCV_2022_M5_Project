from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from utils import show_img
import os, cv2, random
from dataset_dict import get_KITTI_dataset
from detectron2.structures import Instances


""" FEATURE  INFERENCE using MASK R-CNN
    reference: The Elephant in the room https://arxiv.org/pdf/1808.03305.pdf fig.3."""


# Folders
dataset_dir = '/home/group01/MCV_2022_M5_Project/w3/datasets/coco/images/val2017/'
results_dir = './results/task_c/'
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


if __name__ == '__main__':

    model_yalm = ['COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    cfg = get_cfg() 

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_yalm))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yalm)
    predictor = DefaultPredictor(cfg)

    # Run inference with pre-trained Mask R-CNN
    file =  '000000020247.jpg'
    img_path = os.path.join(dataset_dir,file)
    out_path2 = os.path.join(results_dir, file)

    img = inference(img_path)

    cv2.imwrite(out_path2, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    print(f"Feature inference to image: {img_path}")

    




