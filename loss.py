import torch
from torch.nn import MSELoss
from models import VGGFeature
from data_utils import vgg_preprocessing

def gram_matrix(y):
    B, C, H, W = y.size()
    features = y.view(B, C, W*H)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t) / (C*H*W)
    return gram


def loss_function(content_weight, style_weight, yc, ys, y_hat):
	vgg = models.VGGFeature()
	vgg.load_state_dict(torch.load('vgg16feature.pth'))
	if args.cuda:
	    vgg.cuda()

	vgg_preprocessing(yc)
	vgg_preprocessing(ys)
	feature_c = vgg(yc)
	feature_hat = vgg(y_hat)
	loss_feat = content_weight * MSELoss(feature_hat[2], Variable(feature_c[2].data, requires_grad=False))

	feature_s = vgg(Variable(ys, volatile=True))
	gram_s = [gram_matrix(y) for y in feature_s]
	gram_hat = [gram_matrix(y) for y in feature_hat]
	for m in range(0,len(feature_hat)):
        loss_style += style_weight * MSELoss(gram_hat[m], Variable(gram_s[m].data, requires_grad=False))

    return loss_style + loss_feat
