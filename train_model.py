import os
from fastai.vision.all import *
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]="2"

path = Path('/storage/vskovoroda/Stones/data')
dls = SegmentationDataLoaders.from_label_func(path = path/'images',
            bs = 4,
            batch_tfms=[Normalize.from_stats(*imagenet_stats), aug_transforms],
            fnames = get_image_files(path/'images'),
            label_func = lambda o: path/'annotations'/o.name,
            codes = ['no_asbest', 'asbest'])

print('DLS create!')

model = smp.DeepLabV3Plus(
    encoder_name="resnet34", 
    encoder_weights=None,    
    activation = 'sigmoid',
    in_channels=3,           
    classes=1,                
)

learn = Learner(dls, model, loss_func=DiceLoss())

print('Start fit DeepLabV3Plus')
learn.fit(7)

learn.save('DeepLabV3Plus')

print('DeepLabV3Plus Done!')
