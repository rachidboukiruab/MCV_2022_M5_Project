import pandas as pd
from pathlib import Path
from typing import List, Dict
from pycocotools.mask import decode, toBbox, frPyObjects
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from typing import TypedDict, List
import json

class DatasetSplit(TypedDict):
    """
    A typed dict to represent experiment settings. Types should match those in
    the configuration JSON file used as input parameter.
    """
    val: List
    training: List
    #test_set: List
    

def get_KITTI_dataset(path: Path, part: str) -> List[Dict]:
    
    with open('w2\configs\dataset_split.json') as f:
        exp: DatasetSplit = json.load(f)

    anns = []

    for i in exp[part]:
        root_img_dir = path / part / "image_02" / i


        for seq in root_img_dir.glob("*"):
            sequence = seq.parts[-2]
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

                if not len(frame_gt):
                    continue

                ann = []
                for _, obj_id, class_id, height, width, rle in frame_gt.itertuples(index=False):
                    rle = bytearray(rle, "utf8")
                    rleobj = frPyObjects([rle], height, width)
                    bbox = toBbox(rleobj)
                    ann.append({
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": class_id,
                        "segmentation": rleobj,
                        "keypoints": [],
                        "iscrowd": 0
                    })

                anns.append({
                    "file_name": str(img_path),
                    "height": frame_gt["height"].iloc[0],
                    "width": frame_gt["width"].iloc[0],
                    "image_id": int(f"{sequence}{frame:05}"),
                    "sem_seg": str(path / "instances" / sequence / img_name),
                    "annotations": ann
                })

    return anns





