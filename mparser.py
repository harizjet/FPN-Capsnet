import argparse


argParser = argparse.ArgumentParser()
argParser.add_argument('--model', help='fpn-capsnet or ipyramid-capsnet', choices=['fpn-capsnet', 'ipyramid-capsnet'])
argParser.add_argument('--device', help='cuda or cpu', choices=['cuda', 'cpu'])
argParser.add_argument('--criterion', help='capsule-loss or cross-entropy', choices=['capsule-loss', 'cross-entropy'])
argParser.add_argument('--lr_scheduler', help='exponential, step or none', choices=['exponential', 'step', 'none'])
argParser.add_argument('--lr_decay', help='decay rate for lr (used by lr_scheduler)', type=float, default=1)
argParser.add_argument('--lr_decay_step', help='number of epochs before each decay rate for lr (used by step lr_scheduler)', type=int, default=8)
argParser.add_argument('--epochs', help='number of epochs to train', type=int, default=100)
argParser.add_argument('--lr', help='learning rate for optimizer', type=float, default=0.00001)
argParser.add_argument('--batch_size', help='batch size fo train', type=int, default=32)
