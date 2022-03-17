# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Folders
dataset_dir = '/home/mcv/datasets/KITTI-MOTS/'
# results_dir = '/home/group01/MCV_2022_M5_Project/w2/results/task_c/'
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

    model_list = ['COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml']

    for model_yalm in model_list:

        cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(model_yalm))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yalm)
        predictor = DefaultPredictor(cfg)

        '''img_name = '000000.png'
      img_path = os.path.join(dataset_dir, 'training/image_02/0000/', img_name)
      
      img = inference(img_path)
      
      cv2.imwrite(os.path.join(results_dir, img_name), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])'''

        # root: /home/mcv/datasets/KITTI-MOTS/testing/image_02/00XX
        # file: 000XXX.png
        # out_path: ./results/task_c/KITTI-MOTS/testing/image_02/00XX

        # read img_path -> save out_path2

        # model string Detection/InstanceSegmentation
        model_type = model_yalm.split('/')[0].split('-')[-1]
        # Run inference with pre-trained Faster R-CNN (detection) and Mask R-CNN(detection and segmentation) on all KITTI-MOTS dataset
        for root, dirs, files in os.walk(dataset_dir, topdown=True):

            out_path = os.path.join(results_dir, root.split('datasets/')[1])

            aux = out_path.split('/')
            aux.insert(-1, model_type)
            out_path = '/'.join(aux)
            os.makedirs(out_path, exist_ok=True)

            for file in files:
                img_path = os.path.join(root, file)
                out_path2 = os.path.join(out_path, file)

                img = inference(img_path)

                cv2.imwrite(out_path2, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
