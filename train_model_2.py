import os
from fastai.vision.all import *
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]="2"

path = Path('/storage/vskovoroda/Stones/data')
dls = SegmentationDataLoaders.from_label_func(path = path,
            bs = 5,
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
                            max_zoom = 1.0,
                            pad_mode=PadMode.Zeros
                    )],
            fnames = get_image_files(path/'images'),
            label_func = lambda o: path/'annotations'/o.name,
            codes = ['no_asbest', 'asbest'])

print('DLS create!')

# model = smp.UnetPlusPlus(
#     encoder_name="resnet152", 
#     encoder_weights=None,    
#     in_channels=3,
#     activation = 'sigmoid',           
#     classes=1,                
# )

learn = unet_learner(dls, resnet152, loss_func=DiceLoss(), y_range=(0,1),
                pretrained = False)

print('Start fit unet_learner_res152')
learn.fit(7)

learn.save('unet_learner_res152')

print('unet_learner_res152 Done!')
