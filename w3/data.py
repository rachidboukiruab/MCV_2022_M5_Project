from pathlib import Path
from typing import List, Dict

from PIL import Image


def dataset_from_image_dir(path: Path) -> List[Dict]:
    """
    Create a list of dict from an image path in order to perform inference.

    Parameters
    ----------
    path: Path
        Path to the folder where images are stored.

    Returns
    -------
    List[Dict]
        A list of dictionaries in Detectron Format (ready to register).
    """
    output = []
    input_files = [x for x in path.glob("*") if x.is_file()]
    input_files.sort(key=str)

    for ii, img_path in input_files:
        img = Image.open(img_path)

        output.append({
            "file_name": str(img_path),
            "height": img.height,
            "width": img.width,
            "image_id": ii,
        })

    return output
