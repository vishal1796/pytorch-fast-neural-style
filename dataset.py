from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
from os import listdir
from os.path import join
import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img


transform = transforms.Compose([transforms.Scale(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.mul(255))])


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        return input

    def __len__(self):
        return len(self.image_filenames)



def get_training_set(root_dir):
    train_dir = os.join(root_dir, "train")
    dataset_loader = DatasetFromFolder()
    return dataset_loader(train_dir, input_transform=transform)