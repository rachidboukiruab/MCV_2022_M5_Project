import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Dict
from pycocotools.mask import toBbox, frPyObjects, decode
from detectron2.structures import BoxMode
import cv2
from typing_extensions import TypedDict
import json


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
                mask = {
                    "counts": rle.encode('utf8'),
                    "size": [height, width],
                }

                bbox = toBbox(mask).tolist()

                ann.append({
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    "segmentation": mask,
                    "keypoints": [],
                    "iscrowd": 0
                })

            anns.append({
                "file_name": str(img_path),
                "height": frame_gt.iloc[0]["height"],
                "width": frame_gt.iloc[0]["width"],
                "image_id": int(f"{sequence}{frame:05}"),
                "sem_seg_file_name": str(path / "instances" / sequence / img_name),
                "annotations": ann
            })

    return anns


def get_KITTI_dataset_noignore(path: Path, part: str) -> List[Dict]:
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
                if class_id == 10:
                    continue

                # reads rle and decodes it with cocotools
                mask = {
                    "counts": rle.encode('utf8'),
                    "size": [height, width],
                }

                bbox = toBbox(mask).tolist()

                ann.append({
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                    "segmentation": mask,
                    "keypoints": [],
                    "iscrowd": 0
                })

            anns.append({
                "file_name": str(img_path),
                "height": frame_gt.iloc[0]["height"],
                "width": frame_gt.iloc[0]["width"],
                "image_id": int(f"{sequence}{frame:05}"),
                "sem_seg_file_name": str(path / "instances" / sequence / img_name),
                "annotations": ann
            })

    return anns


def get_KITTI_dataset_COCO_ids(path: Path, part: str) -> List[Dict]:
    COCO_classes = {
        0: 50,              # Background anywhere
        1: 2,               # Car to Car
        2: 0,               # Pedestrian to Person
        10: 71              # Ignore to Toaster bc wtf not
    }
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
                mask = {
                    "counts": rle.encode('utf8'),
                    "size": [height, width],
                }

                bbox = toBbox(mask).tolist()

                ann.append({
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": COCO_classes[class_id],
                    "segmentation": mask,
                    "keypoints": [],
                    "iscrowd": 0
                })

            anns.append({
                "file_name": str(img_path),
                "height": frame_gt.iloc[0]["height"],
                "width": frame_gt.iloc[0]["width"],
                "image_id": int(f"{sequence}{frame:05}"),
                "sem_seg_file_name": str(path / "instances" / sequence / img_name),
                "annotations": ann
            })

    return anns
