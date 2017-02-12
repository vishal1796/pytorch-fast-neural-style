from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import FastStyleNet
from data import get_training_set
from data_util import batch_rgb_to_bgr
from loss import loss_function

parser = argparse.ArgumentParser(description='Fast Neural style transfer with PyTorch.')
parser.add_argument('style_image_path', metavar='ref', type=str, help='Path to the style reference image.')
parser.add_argument("data_path", type=str, help="Path to training images")
parser.add_argument("--content_weight", type=float, default=10., help='Content weight')
parser.add_argument("--style_weight", type=float, default=1., help='Style weight')
parser.add_argument("--tv_weight", type=float, default=8.5e-5, help='Total Variation Weight')
parser.add_argument("--image_size", dest="img_size", default=256, type=int, help='Output Image size')
parser.add_argument("--epochs", default=2, type=int, help='Number of epochs')
parser.add_argument("--batchSize", default=4, type=int, help='Number of images per epoch')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


print('===> Loading datasets')
train_set = get_training_set()
data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


print('===> Building model')
model = models.FastStyleNet()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

style = tensor_load_rgbimage(args.style_image, args.style_size)
style = style.repeat(args.batch_size, 1, 1, 1)
style = batch_rgb_to_bgr(style)
if args.cuda:
    style = style.cuda()
xs = Variable(style, volatile=True)


print('===> Training model')
for epoch in range(args.epochs):
    for batch in data_loader:
        data = batch[0].clone()
        data = batch_rgb_to_bgr(data)
        if args.cuda:
            data = data.cuda()
        xc = Variable(data.clone())
        y_hat = model(xc)
        optimizer.zero_grad()
        loss = loss_function(args.content_weight, args.style_weight, yc, ys, y_hat)
        loss.backward()
        optimizer.step()
        print('===> Epoch[{}] batch({}/{}): Loss: {:.4f}'.format(epoch, i, n_iter, loss.data[0]))
    torch.save(model.state_dict(), 'model_{}.pth'.format(epoch))
torch.save(model.state_dict(), 'model.pth')
