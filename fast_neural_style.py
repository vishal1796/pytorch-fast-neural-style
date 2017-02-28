from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from models import ImageTransformNet
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Fast Style Transfer')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_name', default='styleTransfer.jpg', type=str, help='location to save the output image')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
args = parser.parse_args()

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = models.ImageTransformNet()
model.load_state_dict(torch.load(args.model))
if cuda:
    model.cuda()

# load image
img = Image.open(args.input)
img = np.array(img)
img = np.array(img[..., ::-1])  	  # RGB -> BGR
img = img.transpose(2, 0, 1)  		  # (H, W, C) -> (C, H, W)
img = img.reshape((1, ) + img.shape)  # (C, H, W) -> (B, C, H, W)
img = torch.from_numpy(img).float()
img = Variable(img, volatile=True)
if cuda:
    img = img.cuda()

model.eval()
output_img = model(img)

# save output
output_img = output_img.data.cpu().clamp(0, 255).byte().numpy()
output_img = output_img[0].transpose((1, 2, 0))
output_img = output_img[..., ::-1]
output_img = Image.fromarray(output_img)
output_img.save(args.output_name)