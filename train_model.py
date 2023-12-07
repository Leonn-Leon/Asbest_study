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
# nohup python3.9 train_model.py --save_fname=Unet_res152 --encoder=resnet152 --gpu=1 --bs=10 > output2/Unet_res152.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_dense161 --encoder=densenet161 --gpu=2 --bs=10 > output2/Unet_dense161.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_xcep --encoder=xception --gpu=3 --bs=10 > output2/Unet_xcep.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_incv4 --encoder=inceptionv4 --gpu=0 --bs=10 > output2/Unet_incv4.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_vgg19 --encoder=vgg19_bn --gpu=0 --bs=10 > output2/Unet_vgg19.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_effb7 --encoder=efficientnet-b7 --gpu=1 --bs=7 > output2/Unet_effb7.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_dpn131 --encoder=dpn131 --gpu=2 --bs=10 > output2/Unet_dpn131.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_mitb5 --encoder=mit_b5 --gpu=3 --bs=10 > output2/Unet_mitb5.out & f+
# nohup python3.9 train_model.py --save_fname=Unet_mobs4 --encoder=mobileone_s4 --gpu=3 --bs=10 > output2/Unet_mobs4.out & f+

# nohup python3.9 train_model.py --save_fname=MAnet_res152 --encoder=resnet152 --gpu=0 --bs=10 > output2/MAnet_res152.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_dense161 --encoder=densenet161 --gpu=1 --bs=10 > output2/MAnet_dense161.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_xcep --encoder=xception --gpu=2 --bs=10 > output2/MAnets_xcep.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_incv4 --encoder=inceptionv4 --gpu=0 --bs=10 > output2/MAnets_incv4.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_vgg19 --encoder=vgg19_bn --gpu=2 --bs=10 > output2/MAnet_vgg19.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_effb7 --encoder=efficientnet-b7 --gpu=3 --bs=7 > output2/MAnet_effb7.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_dpn131 --encoder=dpn131 --gpu=2 --bs=7 > output2/MAnet_dpn131.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_mitb5 --encoder=mit_b5 --gpu=0 --bs=10 > output2/MAnet_mitb5.out & f+
# nohup python3.9 train_model.py --save_fname=MAnet_mobs4 --encoder=mobileone_s4 --gpu=1 --bs=10 > output2/MAnet_mobs4.out & f+

# nohup python3.9 train_model.py --save_fname=Linknet_res152 --encoder=resnet152 --gpu=0 --bs=10 > output2/Linknet_res152.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_dense161 --encoder=densenet161 --gpu=1 --bs=10 > output2/Linknet_dense161.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_xcep --encoder=xception --gpu=3 --bs=10 > output2/Linknet_xcep.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_incv4 --encoder=inceptionv4 --gpu=0 --bs=10 > output2/Linknet_incv4.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_vgg19 --encoder=vgg19_bn --gpu=1 --bs=10 > output2/Linknet_vgg19.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_effb7 --encoder=efficientnet-b7 --gpu=3 --bs=7 > output2/Linknet_effb7.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_dpn131 --encoder=dpn131 --gpu=2 --bs=10 > output2/Linknet_dpn131.out & f+
# nohup python3.9 train_model.py --save_fname=Linknet_mitb5 --encoder=mit_b5 --gpu=3 --bs=10 > output2/Linknet_mitb5.out & f-
# nohup python3.9 train_model.py --save_fname=Linknet_mobs4 --encoder=mobileone_s4 --gpu=3 --bs=15 > output2/Linknet_mobs4.out & f+

# nohup python3.9 train_model.py --save_fname=PSPNet_res152 --encoder=resnet152 --gpu=1 --bs=10 > output2/PSPNet_res152.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_dense161 --encoder=densenet161 --gpu=1 --bs=10 > output2/PSPNet_dense161.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_xcep --encoder=xception --gpu=0 --bs=10 > output2/PSPNet_xcep.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_incv4 --encoder=inceptionv4 --gpu=1 --bs=20 > output2/PSPNet_incv4.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_vgg19 --encoder=vgg19_bn --gpu=2 --bs=15 > output2/PSPNet_vgg19.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_effb7 --encoder=efficientnet-b7 --gpu=3 --bs=7 > output2/PSPNet_effb7.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_dpn131 --encoder=dpn131 --gpu=0 --bs=10 > output2/PSPNet_dpn131.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_mitb5 --encoder=mit_b5 --gpu=1 --bs=10 > output2/PSPNet_mitb5.out & f+
# nohup python3.9 train_model.py --save_fname=PSPNet_mobs4 --encoder=mobileone_s4 --gpu=3 --bs=20 > output2/PSPNet_mobs4.out & f+

# nohup python3.9 train_model.py --save_fname=FPN_res152 --encoder=resnet152 --gpu=1 --bs=10 > output2/FPN_res152.out & s+
# nohup python3.9 train_model.py --save_fname=FPN_dense161 --encoder=densenet161 --gpu=2 --bs=10 > output2/FPN_dense161.out & s+
# nohup python3.9 train_model.py --save_fname=FPN_xcep --encoder=xception --gpu=0 --bs=10 > output2/FPN_xcep.out & s+
# nohup python3.9 train_model.py --save_fname=FPN_incv4 --encoder=inceptionv4 --gpu=3 --bs=20 > output2/FPN_incv4.out & s+
# nohup python3.9 train_model.py --save_fname=FPN_vgg19 --encoder=vgg19_bn --gpu=2 --bs=15 > output2/FPN_vgg19.out &
# nohup python3.9 train_model.py --save_fname=FPN_effb7 --encoder=efficientnet-b7 --gpu=3 --bs=7 > output2/FPN_effb7.out &
# nohup python3.9 train_model.py --save_fname=FPN_dpn131 --encoder=dpn131 --gpu=0 --bs=10 > output2/FPN_dpn131.out &
# nohup python3.9 train_model.py --save_fname=FPN_mitb5 --encoder=mit_b5 --gpu=1 --bs=10 > output2/FPN_mitb5.out &
# nohup python3.9 train_model.py --save_fname=FPN_mobs4 --encoder=mobileone_s4 --gpu=3 --bs=20 > output2/FPN_mobs4.out &

# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_res152 --encoder=resnet152 --gpu=1 --bs=7 > output2/UnetPlusPlus_res152.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_dense161 --encoder=densenet161 --gpu=2 --bs=7 > output2/UnetPlusPlus_dense161.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_xcep --encoder=xception --gpu=3 --bs=7 > output2/UnetPlusPlus_xcep.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_incv4 --encoder=inceptionv4 --gpu=1 --bs=30 > output2/UnetPlusPlus_incv4.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_vgg19 --encoder=vgg19_bn --gpu=2 --bs=30 > output2/UnetPlusPlus_vgg19.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_effb7 --encoder=efficientnet-b7 --gpu=3 --bs=15 > output2/UnetPlusPlus_effb7.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_dpn131 --encoder=dpn131 --gpu=1 --bs=15 > output2/UnetPlusPlus_dpn131.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_mitb5 --encoder=mit_b5 --gpu=2 --bs=40 > output2/UnetPlusPlus_mitb5.out &
# nohup python3.9 train_model.py --save_fname=UnetPlusPlus_mobs4 --encoder=mobileone_s4 --gpu=3 --bs=50 > output2/UnetPlusPlus_mobs4.out &

# resnet152 - 50(17)
# densenet161 - 50(22)
# xception - 50(14)
# inceptionv4 - 50(13)
# vgg19_bn - 50(16)
# efficientnet-b7 - 30(25.9)
# dpn131 - 30(19)
# mit_b5 - 40(20.8)
# mobileone_s4 - 50(14)

# smp.Unet
# smp.UnetPlusPlus
# smp.MAnet
# smp.Linknet
# smp.PSPNet
# smp.FPN
# smp.PAN
# smp.DeepLabV3
# smp.DeepLabV3Plus
#####

args={}
for i in sys.argv[1:]:
    if i[:2] == '--':
        args[i.split('=')[0][2:]] = i.split('=')[1]
    
save_fname = args['save_fname']
enc_name = args['encoder']
print(f'Models will save to {save_fname}_x')
print(f'Use {enc_name} encoder')

##################
os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu']

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
    
class Square_TargetByPredict(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.name = "Square_Target/Predict"
        super(Square_TargetByPredict, self).__init__()
    
    def forward(self, inputs, targets):
        
        inputs = inputs.view(-1).round()
        targets = targets.view(-1).to(torch.float32).round()
        return inputs.sum()/targets.sum()

class Accuracy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.name = "Accuracy"
        super(Accuracy, self).__init__()
    
    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        
        tp, fp, fn, tn = smp.metrics.get_stats(inputs[:, 0, ...], targets, mode='binary', threshold=0.5)
        # iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        return accuracy

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.name = "IoU"
        super(IoU, self).__init__()
    
    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        
        tp, fp, fn, tn = smp.metrics.get_stats(inputs[:, 0, ...], targets, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        return iou_score

print('Start training...')
ind = 0
for train_idx, val_idx in kf.split(get_image_files(data_path/'images')):
    print('-'*10)
    print(ind)
    print('-'*10)
    dblock = DataBlock(blocks=(ImageBlock(PILImage), MaskBlock(codes=['no_asbest','asbest'])),
                   splitter=IndexSplitter(val_idx),
                   get_y=lambda o: data_path/'annotations'/o.name,
                   item_tfms=[Resize(1000), RandomCrop(512)],
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
                                    bs=int(args['bs']))
    model = smp.FPN(
        encoder_name = enc_name, 
        encoder_weights = None,    
        in_channels = 3,    
        activation = 'sigmoid',       
        classes = 1,                
    )

    print('DLS create!')
    # sched = {'lr': combined_cos(0.2,0.1,1.,0.001)}
    learn = Learner(dls, model, loss_func=BCEnDiceLoss(), opt_func=Adam,
            metrics=[smp.losses.JaccardLoss(mode='binary', from_logits=False),
                     Accuracy(),
                     IoU(),
                     smp.losses.FocalLoss(mode='binary'),
                     smp.losses.LovaszLoss(mode='binary', from_logits=False),
                     smp.losses.MCCLoss(),
                     smp.losses.TverskyLoss(mode='binary', log_loss=False, from_logits=False),
                     smp.losses.DiceLoss(mode='binary', log_loss=False, from_logits=False),
                     Square_TargetByPredict()
                    ],
                cbs = [
                       EarlyStoppingCallback(patience=8),
                       SaveModelCallback(monitor='valid_loss', fname = save_fname+'_'+str(ind))
                       ])
    with learn.no_bar():
        learn.fit(90)
        
    np.save('losses2/'+save_fname+'_'+str(ind)+'.npy', np.array(learn.recorder.losses))
    
    torch.cuda.empty_cache()
    
    ind += 1

print(save_fname+' Done!')