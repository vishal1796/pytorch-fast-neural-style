from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from models import FastStyleNet
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Fast Style Transfer')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', default='styleTransfer.jpg', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

model = models.FastStyleNet()
model.load_state_dict(torch.load(args.model))
if args.cuda:
    model.cuda()

# load image
img = Image.open(args.input)
img = np.array(img)  # PIL->numpy
img = np.array(img[..., ::-1])  # RGB->BGR
img = img.transpose(2, 0, 1)  # HWC->CHW
img = img.reshape((1, ) + img.shape)  # CHW->BCHW
img = torch.from_numpy(img).float()
img = Variable(img, volatile=True)
if args.cuda:
    img = img.cuda()

model.eval()
output = model(img)

# save output
output = output.data.cpu().clamp(0, 255).byte().numpy()
output = output[0].transpose((1, 2, 0))
output = output[..., ::-1]
output = Image.fromarray(output)
output.save(args.output)