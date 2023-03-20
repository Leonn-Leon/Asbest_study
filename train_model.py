import matplotlib.pyplot
from fastai.vision.all import *
import os
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp

#####
# nohup python3.9 train_model.py Unet++_xcep xception > output/Unet++_xcep.out &
#####

save_fname = sys.argv[1]
enc_name = sys.argv[2]
print(f'Models will save to {save_fname}_x')
print(f'Use {enc_name} encoder')

##################
os.environ["CUDA_VISIBLE_DEVICES"]="1"

kf = KFold(n_splits=5, shuffle=True)

data_path = Path('/storage/vskovoroda/Stones/data')

class BCEnDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        __name__ = 'BCE_Dice'
        super(BCEnDiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=0.0):
        
        # inputs = F.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1).to(torch.float32)
        
        intersection = (inputs * targets).sum()
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        bse = F.binary_cross_entropy(inputs, targets)
        return bse*.5 + dice*.5

print('Start training...')
ind = 0
for train_idx, val_idx in kf.split(get_image_files(data_path/'images')):
    print('-'*10)
    print(ind)
    print('-'*10)
    dblock = DataBlock(blocks=(ImageBlock(PILImage), MaskBlock(codes=['no_asbest','asbest'])),
                   splitter=IndexSplitter(val_idx),
                   get_y=lambda o: data_path/'annotations'/o.name,
                   item_tfms=[Resize(1000), RandomCrop(256)],
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
                                    pad_mode=PadMode.Reflection
                    )])
    dls = SegmentationDataLoaders.from_dblock(dblock,
                                    get_image_files(data_path/'images'),
                                    path=data_path,
                                    bs = 30)
    model = smp.UnetPlusPlus(
        encoder_name = enc_name, 
        encoder_weights = None,    
        in_channels = 3,    
        activation = 'sigmoid',       
        classes = 1,                
    )

    print('DLS create!')

    learn = Learner(dls, model, loss_func=BCEnDiceLoss(),
                metrics=[smp.losses.JaccardLoss(mode='binary'),
                         smp.losses.FocalLoss(mode='binary'),
                         smp.losses.LovaszLoss(mode='binary'),
                         smp.losses.MCCLoss()
                        ],
                cbs = [EarlyStoppingCallback(patience=8),
                       SaveModelCallback(monitor='valid_loss', fname = save_fname+'_'+str(ind)),
                        ])
    with learn.no_bar():
        learn.fit(90)
        
    np.save('losses/'+save_fname+'_'+str(ind)+'.npy', np.array(learn.recorder.losses))
    
    torch.cuda.empty_cache()
    
    ind += 1

print('UnetPlusPlus Done!')