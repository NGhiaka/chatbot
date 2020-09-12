#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.DATA_PATH           = "./data/cafe_chatbot/conversation.csv"
__C.TRAIN.BATCH_SIZE          = 8
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.BUFFER_SIZE         = 20000
__C.TRAIN.MAX_LENGTH          = 40
__C.TRAIN.MAX_SAMPLES         = 50000
__C.TRAIN.EPOCHS              = 200

__C.TRAIN.CHECKPOINT_PATH     = "./checkpoints/train"
__C.TRAIN.NUM_LAYERS          = 2
__C.TRAIN.D_MODEL             = 256
__C.TRAIN.NUM_HEADS           = 8
__C.TRAIN.UNITS               = 512
__C.TRAIN.DROPOUT             = 0.1

