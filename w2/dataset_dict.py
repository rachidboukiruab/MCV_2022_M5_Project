from detectron2.structures import BoxMode
import os
import cv2
import numpy as np

def get_class_name(class_id):
    classes = {
      0:'Car',
      1:'Van',
      2:'Truck',
      3:'Pedestrian',
      4:'Person_sitting',
      5:'Cyclist',
      6:'Tram',
      7:'Misc',
      8:'DontCare'  
    }

    return classes.get(class_id)

def get_KITTI_MOTS_dicts(dataset_dir):

    instances = dataset_dir+'/instances'

    dataset_dicts = []
    for (root,dirs,files) in os.walk(instances, topdown=True):
        record = {}
        
        filename = files
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        obj_ids = np.unique(img)
        # to correctly interpret the id of a single object
        obj_id = obj_ids[0]
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000
        
        record["file_name"] = filename
        record["image_id"] = class_id
        record["height"] = height
        record["width"] = width
        
      
"""         annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train") """