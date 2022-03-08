import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.datasets import ImageFolder

DATASET_PATH = Path("/home/pau/Documents/master/M3/project/MIT_split")

transfs = transforms.Compose([
    transforms.ToTensor()
])

test_data = ImageFolder(str(DATASET_PATH / "test"), transform=transfs)
test_dataloader = DataLoader(
    test_data, 2, False
)

for ii, (images, labels) in enumerate(test_dataloader):
    plt.figure()
    plt.imshow(images[0].permute((1, 2, 0)).numpy())
    plt.show()
    plt.close()

    break
