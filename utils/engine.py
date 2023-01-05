import os
import json
import csv
import logging
import sys
import time
import re
import numpy as np 

from utils.utils import accuracy

import torch
from torchsummary import summary
from torch.utils.data import DataLoader


class Engine(object):
    """
    Main engine to train and save model
    Example:
    $ engine = Engine(basedir=PATH, model=model, epochs=10, batch_size=8, learning_rate=1e-4, optimizer=optimizer, criterion=criterion, train_dataset=train_dataset, val_dataset=val_dataset, use_cuda=False))
    $ engine.train()
    """
    def __init__(self, basedir, model, epochs, batch_size, learning_rate, optimizer, criterion, train_dataset, val_dataset, device=torch.device('cpu')):    
        self.basedir = basedir
        self.model = model

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.foldername = self._getFolderName()
        self.device = device

    def _initLog(self):
        self.logger = logging.getLogger(self.foldername)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s :: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

        file_handler = logging.FileHandler(os.path.join(self.basedir, self.foldername, 'training.log'))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

    def _stopLog(self):
        self.logger.disabled = True
    
    def _getFolderName(self):
        tmp = [False] * 998
        for file in os.listdir(self.basedir):
            if not re.search('^\d{3}$', file):
                continue
            tmp[int(file)-1] = True

        filei = None
        for i in range(998):
            if not tmp[i]:
                filei = i + 1
                break

        filea = [0] * 3
        for i in range(2, -1, -1):
            filea[i] = str(filei % 10)
            filei //= 10
            
        return ''.join(filea)
    
    def _getDataLoader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def _createFolder(self):
        os.makedirs(os.path.join(self.basedir, self.foldername))
    
    def _saveSummaryModel(self):
        with open(os.path.join(self.basedir, self.foldername, 'summary.txt'), 'w+', encoding="utf-8") as f:
            f.write(str(summary(self.model)))

    def _saveConfig(self):
        config = {}

        config['epochs'] = self.epochs
        config['batch_size'] = self.batch_size
        config['learning_rate'] = self.learning_rate
        config['optimizer'] = str(self.optimizer)
        config['criterion'] = str(self.criterion)

        with open(os.path.join(self.basedir, self.foldername, 'config.json'), 'w+') as f:
            json.dump(config, f, indent=6)

    def _saveAccuracy(self):
        with open(os.path.join(self.basedir, self.foldername, 'accuracy.csv'), 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerow(['epoch', 'accuracy'])
            for i in range(len(self.accuracy)):
                writer.writerow([i+1, self.accuracy[i]])
    
    def _saveLoss(self):
        with open(os.path.join(self.basedir, self.foldername, 'loss.csv'), 'w+') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerow(['epoch', 'loss'])
            for i in range(len(self.loss)):
                writer.writerow([i+1, self.loss[i]])

    def _saveModel(self):
        torch.save(self.model, os.path.join(self.basedir, self.foldername, 'model.pb'))
    
    def train(self):
        self._createFolder()
        self._initLog()
        self._getDataLoader()
        self.accuracy = []
        self.loss = []

        TIME_NOW = time.time()

        for epoch in range(1, self.epochs+1):
            self.logger.info(f'START EPOCH - {epoch}')

            running_loss = 0
            total_loss = 0

            N1 = len(self.train_loader)

            self.model.train(True)
            for i, sample in enumerate(self.train_loader):
                x, y = sample

                x = x.to(self.device)
                y = y.to(self.device)

                ypred =self.model(x)

                loss = self.criterion(y, ypred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                if i % 5 == 4:
                    self.logger.info(f'EPOCH: [{epoch:3d},{self.epochs:3d}]   | MINI-BATCH: [{i + 1:4d},{N1:4d}]   | LOSS: {running_loss / 5:.4f}   --  {time.time()-TIME_NOW:.2f}s')
                    running_loss = 0

            self.loss.append(total_loss / N1)

            val_acc = []
            val_loss = []
            self.model.train(False)
            for sample in self.val_loader:
                x, y = sample

                x = x.to(self.device)
                y = y.to(self.device)

                ypred = self.model(x)

                loss = self.criterion(y, ypred)
                acc = accuracy(y, ypred)

                val_loss.append(loss.item())
                val_acc.append(acc)

            self.accuracy.append(np.mean(val_acc))
            self.logger.info(f'EPOCH: [{epoch:3d},{self.epochs:3d}]   | VAL-ACCURACY: {np.mean(val_acc):.4f}      | VAL-LOSS: {np.mean(val_loss):.4f}   --  {time.time()-TIME_NOW:.2f}s')

        self._saveSummaryModel()
        self._saveConfig()
        self._saveAccuracy()
        self._saveLoss()
        self._saveModel()
        self._stopLog()
