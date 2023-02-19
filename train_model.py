import os
from fastai.vision.all import *
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]="2"

path = Path('/storage/vskovoroda/Stones/data')
dls = SegmentationDataLoaders.from_label_func(path = path,
            bs = 2,
            item_tfms = RandomCrop(480),
            batch_tfms=[Normalize.from_stats(*imagenet_stats), 
                        *aug_transforms(
                            mult=1.0,
                            do_flip=True,
                            flip_vert=True,
                            max_rotate=45.0,
                            max_lighting=0.0,
                            max_warp=0.0,
                            p_affine=0.8,
                            pad_mode=PadMode.Zeros
                    )],
            fnames = get_image_files(path/'images'),
            label_func = lambda o: path/'annotations'/o.name,
            codes = ['no_asbest', 'asbest'])

print('DLS create!')

model = smp.UnetPlusPlus(
    encoder_name="resnext101_32x48d", 
    encoder_weights=None,    
    in_channels=3,    
    activation = 'sigmoid',       
    classes=1,                
)

learn = Learner(dls, model, loss_func=DiceLoss(),
                metrics=[smp.losses.TverskyLoss(mode='binary'),
                     smp.losses.JaccardLoss(mode='binary'),
                     smp.losses.FocalLoss(mode='binary'),
                     smp.losses.LovaszLoss(mode='binary'),
                     smp.losses.MCCLoss()
                    ])

print('Start fit UnetPlusPlus')
learn.fit(5)

learn.save('UnetPlusPlus')

print('Start fine_tune')
learn.fine_tune(2)

print('UnetPlusPlus Done!')
