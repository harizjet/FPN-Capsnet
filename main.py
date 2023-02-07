from mparser import argParser

from lib.model.ipyramid_capsnet import IPyramidCapsuleNet
from lib.model.fpn_capsnet import FPN101CapsuleNet

from utils.loss import CapsuleLoss, CrossEntropyLoss
from utils.engine import Engine, TestEngine
from utils.dataset import MNISTdataset
from utils.transform import *

import torch
from torchvision import transforms
from torch.optim import Adam


"""
Example:
python main.py \
    --model fpn-capsnet \
    --device cuda \
    --criterion capsule-loss \
    --lr_scheduler step \
    --lr_decay 0.98 \
    --lr_decay_step 8 \
    --epochs 1 \
    --lr 0.00001 \
    --batch_size 32
"""

if __name__ == "__main__":
    arg = argParser.parse_args()
    
    MODEL_NAME = arg.model
    DEVICE = torch.device(arg.device)
    CRITERION_NAME = arg.criterion
    LR_SCHEDULER_NAME = arg.lr_scheduler
    LR_DECAY = arg.lr_decay
    LR_DECAY_STEP = arg.lr_decay_step
    EPOCHS = arg.epochs
    LEARNING_RATE = arg.lr
    BATCHSIZE = arg.batch_size
    MODEL_DIR = "./src/model/"
    TEST_DIR = "./src/output/"
    DATA_DIR = "./data/mnist/"

    match MODEL_NAME:
        case "fpn-capsnet":
            MODEL = IPyramidCapsuleNet(DEVICE).to(DEVICE)
        case "ipyramid-capsnet":
            MODEL = FPN101CapsuleNet(DEVICE).to(DEVICE)

    match CRITERION_NAME:
        case "capsule-loss":
            CRITERION = CapsuleLoss()
        case "cross-entropy":
            CRITERION = CrossEntropyLoss()
    
    OPTIMIZER = Adam(MODEL.parameters(), lr=LEARNING_RATE)

    match LR_SCHEDULER_NAME:
        case "exponential":
            LR_SCHEDULER = torch.optim.lr_scheduler.ExponentialLR(optimizer=OPTIMIZER, gamma=LR_DECAY)
        case "step":
            LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(optimizer=OPTIMIZER, gamma=LR_DECAY, step_size=LR_DECAY_STEP)
        case "none":
            LR_SCHEDULER = None

    # full-train
    transform_x = transforms.Compose([toTensor(type=torch.float32), augmentation((28, 28), isBatch=True)])
    transform_y = transforms.Compose([toTensor(type=torch.int32)])
    DATASET = MNISTdataset(isTrain=True, transform_x=transform_x, transform_y=transform_y, data_dir=DATA_DIR)
    engine = Engine(basedir=MODEL_DIR, model=MODEL, epochs=EPOCHS, batch_size=BATCHSIZE, 
                    learning_rate=LEARNING_RATE, optimizer=OPTIMIZER, lr_scheduler=LR_SCHEDULER, 
                    criterion=CRITERION, train_dataset=DATASET, device=DEVICE)
    engine.train(valid=False)
    
    # full-test
    transform_x = transforms.Compose([toTensor(type=torch.float32)])
    DATASET = MNISTdataset(isTrain=False, transform_x=transform_x, data_dir=DATA_DIR)
    engine = TestEngine(basedir=TEST_DIR, model=MODEL, dataset=DATASET, device=DEVICE, model_name=engine.foldername)
    engine.test()
