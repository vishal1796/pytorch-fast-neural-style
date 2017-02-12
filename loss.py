import torch
from torch.nn import MSELoss

def gram_matrix(y):
    B, C, H, W = y.size()
    features = y.view(B, C, W*H)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t) / (C*H*W)
    return gram


def FeatureLoss(content_weight, yc, y_hat):
	# mean sbtraction
	feature_c = vgg(yc)
	feature_hat = vgg(y_hat)
	loss_feat = content_weight * MSELoss(feature_hat[2], Variable(feature_c[2].data, requires_grad=False))
	return loss_feat


def StyleLoss(style_weight, ys, y_hat):
	# mean sbtraction
	feature_s = vgg(Variable(ys, volatile=True))
	gram_s = [gram_matrix(y) for y in feature_s]
	feature_hat = vgg(y_hat)
	gram_hat = [gram_matrix(y) for y in feature_hat]
	for m in range(0,len(feature_hat)):
        loss_style += style_weight * MSELoss(gram_hat[m], Variable(gram_s[m].data, requires_grad=False))
    return loss_style
