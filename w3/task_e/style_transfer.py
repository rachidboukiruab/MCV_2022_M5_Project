import os
from os import listdir
from os.path import isfile, join

import torchvision.models as models
from torchvision.utils import save_image

from utils import image_loader, make_dirs, device, run_style_transfer, cnn_normalization_mean, cnn_normalization_std

result_path = '/home/group01/MCV_2022_M5_Project/w3/te_data/results'
style_path = "/home/group01/MCV_2022_M5_Project/w3/te_data/style"
content_path = "/home/group01/MCV_2022_M5_Project/w3/te_data/content"

# Generate a folder to save results
make_dirs(result_path)

style_images = [f for f in listdir(style_path) if isfile(join(style_path, f))]
content_images = [f for f in listdir(content_path) if isfile(join(content_path, f))]

# sort list
style_images = sorted(style_images, key=lambda x: int(os.path.splitext(x)[0]))
content_images = sorted(content_images, key=lambda x: int(os.path.splitext(x)[0]))

cnn = models.vgg19(pretrained=True).features.to(device).eval()

for ii, (style, content) in enumerate(zip(style_images, content_images)):

    style_img = image_loader(style)
    content_img = image_loader(content)

    assert style_img.size() == content_img.size(), "style & content imgs should be same size"

    # input image as initializer
    input_img = content_img
    # uncomment & replace 4 white noise initializer
    """
    input_img = torch.randn(content_img.data.size(), device=device, requires_grad=False)
    # Gaussian smooth

    blur = cv2.GaussianBlur(input_img.detach().squeeze().cpu().numpy(), (3, 3), 0)
    blur = blur - blur.min()
    blur = blur / blur.max()
    blur = torch.from_numpy(blur).to(device).unsqueeze(0)
    """
    step = 900

    for j in range(10):
        input_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                       content_img, style_img, input_img, num_steps=step, print_step=100,
                                       content_weight=1, style_weight=1000000)

    save_image(input_img, os.path.join(result_path, f'{ii}.png'))
