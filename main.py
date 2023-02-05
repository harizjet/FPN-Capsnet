import argparse
from utils.engine import Engine, TestEngine
import torch
from lib.model.ipyramid_capsnet import IPyramidCapsuleNet
from lib.model.fpn_capsnet import FPN101CapsuleNet

argParser = argparse.ArgumentParser()
argParser.add_argument('--model', help='fpn-capsnet or ipyramid-capsnet', choices=['fpn-capsnet', 'ipyramid-capsnet'])
argParser.add_argument('--device', help='cuda or cpu', choices=['cuda', 'cpu'])
argParser.add_argument('--criterion', help='capsule-loss or cross-entropy', choices=['capsule-loss', 'cross-entropy'])
argParser.add_argument('--lr_scheduler', help='exponential, step or none', choices=['exponential', 'step', 'none'])
argParser.add_argument('--lr_decay', help='decay rate for lr (used by lr_scheduler)', type=float, default=1)
argParser.add_argument('--epochs', help='number of epochs to train', type=int, default=100)
argParser.add_argument('--lr', help='learning rate for optimizer', type=float, default=0.00001)
argParser.add_argument('--batch_size', help='batch size fo train', type=int, default=32)


if __name__ == "__main__":
    arg = argParser.parse_args()
    
    MODEL_NAME = arg.model
    DEVICE = torch.device(arg.device)
    CRITERION = arg.criterion
    LR_SCHEDULER = arg.lr_scheduler
    LR_DECAY = arg.lr_decay
    EPOCHS = arg.epochs
    LEARNING_RATE = arg.lr
    BATCHSIZE = arg.batch_size

    match MODEL_NAME:
        case "fpn-capsnet":
            MODEL = IPyramidCapsuleNet(DEVICE).to(DEVICE)
        case "ipyramid-capsnet":
            MODEL = FPN101CapsuleNet(DEVICE).to(DEVICE)

    