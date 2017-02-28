import os
import torch
from torch.nn import MSELoss
from torch.utils.serialization import load_lua
from torch.autograd import Variable
import models
from data_utils import vgg_mean_subtraction


def vgg16_model():
    if not os.path.exists('vgg16feature.pth'):
        if not os.path.exists('vgg16.t7'):
            os.system('wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
        vgglua = load_lua('vgg16.t7')
        vgg = models.VGGFeature()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), 'vgg16feature.pth')

def gram_matrix(y):
    B, C, H, W = y.size()
    features = y.view(B, C, W*H)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t) / (C*H*W)
    return gram

def loss_function(content_weight, style_weight, yc, ys, y_hat, cuda):
    vgg16_model()
    vgg = models.VGGFeature()
    vgg.load_state_dict(torch.load('vgg16feature.pth'))
    criterion = torch.nn.MSELoss()

    if cuda:
        vgg = vgg.cuda()
        criterion = criterion.cuda()

    vgg_mean_subtraction(yc)
    vgg_mean_subtraction(ys)
    vgg_mean_subtraction(y_hat)

    feature_c = vgg(yc)
    feature_hat = vgg(y_hat)
    feat_loss = content_weight * criterion(feature_hat[2], Variable(feature_c[2].data, requires_grad=False))

    feature_s = vgg(ys)
    gram_s = [gram_matrix(y) for y in feature_s]
    gram_hat = [gram_matrix(y) for y in feature_hat]
    style_loss = 0
    for m in range(0, len(feature_hat)):
        style_loss += style_weight * criterion(gram_hat[m], Variable(gram_s[m].data, requires_grad=False))

    return style_loss + feat_loss
