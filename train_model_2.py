import os
from fastai.vision.all import *
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]="3"

path = Path('/storage/vskovoroda/Stones/data')
dls = SegmentationDataLoaders.from_label_func(path = path,
            bs = 1,
            batch_tfms=[Normalize.from_stats(*imagenet_stats), aug_transforms],
            fnames = get_image_files(path/'images'),
            label_func = lambda o: path/'annotations'/o.name,
            codes = ['no_asbest', 'asbest'])

print('DLS create!')

model = smp.UnetPlusPlus(
    encoder_name="resnet34", 
    encoder_weights=None,    
    in_channels=3,
    activation = 'sigmoid',           
    classes=1,                
)

learn = Learner(dls, model, loss_func=DiceLoss())

print('Start fit UnetPlusPlus')
learn.fit(7)

learn.save('UnetPlusPlus')

print('UnetPlusPlus Done!')
