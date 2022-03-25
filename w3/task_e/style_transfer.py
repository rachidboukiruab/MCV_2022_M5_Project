import os
from os import listdir
from os.path import isfile, join

import torchvision.models as models
from torchvision.utils import save_image
from config import RESULT_PATH, STYLE_PATH, CONTENT_PATH
from utils import image_loader, make_dirs, device, run_style_transfer, cnn_normalization_mean, cnn_normalization_std


# Generate a folder to save results
make_dirs(RESULT_PATH)

style_images = [join(STYLE_PATH,f) for f in listdir(STYLE_PATH) if isfile(join(STYLE_PATH, f))]
content_images = [join(CONTENT_PATH,f) for f in listdir(CONTENT_PATH) if isfile(join(CONTENT_PATH, f))]

# sort list
style_images = sorted(style_images, key=lambda x: int(os.path.splitext(x)[0]))
content_images = sorted(content_images, key=lambda x: int(os.path.splitext(x)[0]))

print(style_images)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

for ii, (style, content) in enumerate(zip(style_images, content_images)):
    print(style)

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

    save_image(input_img, os.path.join(RESULT_PATH, f'{ii}.png'))
