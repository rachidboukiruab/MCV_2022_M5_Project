import json
import os
from os import listdir
from os.path import isfile, join

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from config import RESULT_PATH
# run this first only once
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
from utils import make_dirs

INFERENCE_PATH = RESULT_PATH
model = models.resnet50(pretrained=True)
model.eval()

imgs_path = [f for f in listdir(INFERENCE_PATH) if isfile(join(INFERENCE_PATH, f))]
imgs_path = sorted(imgs_path, key=lambda x: int(os.path.splitext(x)[0]))

make_dirs(join(INFERENCE_PATH,"classification"))

results_dict = {}
for filename in imgs_path:
    img_name = filename.split("/")[-1]

    input_image = Image.open(filename)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        results_dict[img_name] = (categories[top5_catid[i]], top5_prob[i].item())

    with open(join(INFERENCE_PATH,"classification",'class_result.json'), 'w') as outfile:
        json.dump(results_dict, outfile)

print("PROCESS FINISHED")