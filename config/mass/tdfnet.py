from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mass import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils


# training hparam
max_epoch = 105
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 0.0025 
backbone_lr = 1e-3
backbone_weight_decay = 0.0025
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "mass-mamba-cnn-v1"
weights_path = "/root/EB-TDFNet/log/mass/{}".format(weights_name)
test_weights_name = "mass-mamba-cnn-v1-best"
log_name = "/root/EB-TDFNet/log/mass/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
# pretrained_ckpt_path = "/root/EB-TDFNet/pre_trained_weights/whu-mamba-cnn-v9165-best4.ckpt"
pretrained_ckpt_path = None
load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth',
resume_ckpt_path = None

from geoseg.models.TDFNet import TDFNet
net = TDFNet(input_channels=3, 
               num_classes=2,
               mid_channel = 96,
               depths=[2, 2, 9, 2], 
               depths_decoder=[2, 9, 2, 2],
               drop_path_rate=0.1,
               load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/vmamba_tiny_e292.pth', 
               pretrained = True)
if load_ckpt_path is not None:
    net.load_from()


# define the loss
loss = EdgeLoss(ignore_index=255)
use_aux_loss = False

# define the dataloader


train_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/train/", mode='train', mosaic_ratio=0.25, transform=get_training_transform())
val_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/val/", mode='val', transform=get_validation_transform())
test_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/test/", mode='val', transform=get_validation_transform())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)