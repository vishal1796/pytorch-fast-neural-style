import os
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
from dataset import DatasetFromFolder

transform = transforms.Compose([transforms.Scale(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.mul(255))])
def get_training_set():
    root_dir = download_bsd300()
    train_dir = os.join(root_dir, "train")
    return DatasetFromFolder(train_dir, input_transform=transform)


def download_convert_vgg16_model():
    if not os.path.exists('vgg16feature.pth'):
        if not os.path.exists('vgg16.t7'):
            os.system('wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
        vgglua = load_lua('vgg16.t7')
        vgg = models.VGGFeature()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), 'vgg16feature.pth')


def vgg_preprocessing(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch -= Variable(mean)


# batch: Bx3xHxW
def batch_rgb_to_bgr(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


# result: RGB CxHxW [0,255] torch.FloatTensor
def load_image(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img
