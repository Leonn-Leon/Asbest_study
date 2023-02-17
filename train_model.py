import os
from fastai.vision.all import *
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def get_msk(o):
    img = cv2.imread(str(data_path/'annotations'/o.name))
    img[img>0] = 1
    return img

path = Path('/storage/vskovoroda/Stones/data')
dls = SegmentationDataLoaders.from_label_func(path = path,
            # item_tfms=Resize(2000),
            bs = 2,
            batch_tfms=[Normalize.from_stats(*imagenet_stats), aug_transforms],
            fnames = get_image_files(path/'images'),
            label_func = lambda o: path/'annotations'/o.name,
            codes = ['no_asbest', 'asbest'])

print('DLS create!')

model = smp.Linknet(
    encoder_name="resnet34", 
    encoder_weights=None,    
    in_channels=3,           
    classes=1,                
)

learn = Learner(dls, model, loss_func=DiceLoss())

print('Start fit model')
learn.fit(5)

learn.save('Linknet')

print('Done!')