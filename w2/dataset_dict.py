import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Dict
from pycocotools.mask import toBbox, frPyObjects
from detectron2.structures import BoxMode
import cv2
from typing_extensions import TypedDict
import json
from pycocotools import _mask as coco_mask
import base64
import zlib


def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation


class DatasetSplit(TypedDict):
    """
    A typed dict to represent experiment settings. Types should match those in
    the configuration JSON file used as input parameter.
    """
    val: List
    training: List
    # test_set: List


def get_KITTI_dataset(path: Path, part: str) -> List[Dict]:
    with open('./configs/dataset_split.json') as f_splits:
        sequences = json.load(f_splits)[part]

    if part == "val":
        part = "training"

    root_img_dir = path / part / "image_02"
    anns = []

    for seq in root_img_dir.glob("*"):
        sequence = seq.parts[-1]

        # Ensure the sequence belongs to the selected partition
        if sequence not in sequences:
            continue

        with open(path / "instances_txt" / (sequence + ".txt")) as f_ann:
            gt = pd.read_table(
                f_ann,
                sep=" ",
                header=0,
                names=["frame", "obj_id", "class_id", "height", "width", "rle"],
                dtype={"frame": int, "obj_id": int, "class_id": int,
                       "height": int, "width": int, "rle": str}
            )

        for img_path in seq.glob("*.png"):
            img_name = img_path.parts[-1]
            frame = int(img_path.parts[-1].split('.')[0])
            frame_gt = (gt[gt["frame"] == frame])

            if len(frame_gt) == 0:
                continue

            ann = []
            for _, obj_id, class_id, height, width, rle in frame_gt.itertuples(index=False):
                # reads rle and decodes it with cocotools
                uncodedStr = base64.b64decode(rle)
                uncompressedStr = zlib.decompress(uncodedStr, wbits=zlib.MAX_WBITS)
                detection = {
                    'size': [width, height],
                    'counts': uncompressedStr
                }
                detlist = []
                detlist.append(detection)
                mask = coco_mask.decode(detlist)
                binaryMask = mask.astype('bool')
                # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py

                bbox = toBbox(rle)
                print(bbox)

                ann.append({
                    "bbox": bbox.flatten(),
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    "segmentation": binaryMask,
                    "iscrowd": 0
                })

            anns.append({
                "file_name": str(img_path),
                "height": frame_gt.iloc[0]["height"],
                "width": frame_gt.iloc[0]["width"],
                "image_id": int(f"{sequence}{frame:05}"),
                "sem_seg": str(path / "instances" / sequence / img_name),
                "annotations": ann
            })

    return anns