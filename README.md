# Pytorch implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

## Requirement
- [Pytorch](http://pytorch.org/)
```
$ conda install pytorch torchvision -c soumith
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py --style_image "images/style/xyz.jpg" --dataset_path "images/content" --cuda
```

## Generate
```
python fast_neural_style.py --input_image "images/input/xyz.jpg" -model "model_epoch_2" --output_name "output.jpg" --cuda
```

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)
