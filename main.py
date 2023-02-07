import argparse
from utils.engine import Engine, TestEngine
import torch
from lib.model.ipyramid_capsnet import IPyramidCapsuleNet
from lib.model.fpn_capsnet import FPN101CapsuleNet
from utils.loss import CapsuleLoss, CrossEntropyLoss
from torch.optim import Adam
from utils.dataset import MNISTdataset
from utils.transform import *
from torchvision import transforms


argParser = argparse.ArgumentParser()
argParser.add_argument('--model', help='fpn-capsnet or ipyramid-capsnet', choices=['fpn-capsnet', 'ipyramid-capsnet'])
argParser.add_argument('--device', help='cuda or cpu', choices=['cuda', 'cpu'])
argParser.add_argument('--criterion', help='capsule-loss or cross-entropy', choices=['capsule-loss', 'cross-entropy'])
argParser.add_argument('--lr_scheduler', help='exponential, step or none', choices=['exponential', 'step', 'none'])
argParser.add_argument('--lr_decay', help='decay rate for lr (used by lr_scheduler)', type=float, default=1)
argParser.add_argument('--lr_decay_step', help='number of epochs before each decay rate for lr (used by step lr_scheduler)', type=float, default=8)
argParser.add_argument('--epochs', help='number of epochs to train', type=int, default=100)
argParser.add_argument('--lr', help='learning rate for optimizer', type=float, default=0.00001)
argParser.add_argument('--batch_size', help='batch size fo train', type=int, default=32)


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
    BASE_DIR = "./src/model/"

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
    transform_x = transforms.Compose([toTensor(), augmentation((28, 28), isBatch=True)])
    transform_y = transforms.Compose([toTensor()])
    DATASET = MNISTdataset(isTrain=True, transform_x=transform_x, transform_y=transform_y)
    engine = Engine(basedir=BASE_DIR, model=MODEL, epochs=EPOCHS, batch_size=BATCHSIZE, 
                    learning_rate=LEARNING_RATE, optimizer=OPTIMIZER, lr_scheduler=LR_SCHEDULER, 
                    criterion=CRITERION, train_dataset=DATASET, device=DEVICE)
    engine.train(valid=False)
    
    # full-test
    transform_x = transforms.Compose([toTensor()])
    transform_y = transforms.Compose([toTensor()])
    DATASET = MNISTdataset(isTrain=False, transform_x=transform_x, transform_y=transform_y)
    engine = TestEngine(basedir=BASE_DIR, model=MODEL, dataset=DATASET, device=DEVICE, model_name=engine.foldername)
    engine.test()
