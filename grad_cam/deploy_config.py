# import tensorflow as tf
import torch
import math
# from loss_funnction_And_matrics import*
###---Number-of-GPU
NUM_OF_GPU = 1
# DISTRIIBUTED_STRATEGY_GPUS=["gpu:0"]
DISTRIIBUTED_STRATEGY_GPUS='cuda:0'
##Network Configuration
NUMBER_OF_CLASSES = 2 #5
INPUT_PATCH_SIZE = (1, 16, 16, 16)#(224,160,160, 1)
TRAIN_NUM_RES_UNIT = 3
TRAIN_CLASSIFY_LEARNING_RATE = 4e-3
# TRAIN_CLASSIFY_LEARNING_RATE =lr#1e-4
# OPTIMIZER = torch.optim.SGD(lr=TRAIN_CLASSIFY_LEARNING_RATE,momentum=0.9,weight_decay=1e-4)
# TRAIN_CLASSIFY_LOSS=Weighted_BCTL
# OPTIMIZER = torch.optim.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,eps=1e-5)
# OPTIMIZER= optimizers #torch.nn.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)

