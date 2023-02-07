import os
import json
import csv
import logging
import sys
import time
import re
import numpy as np 
import shutil

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
    def __init__(self, basedir, model, epochs, batch_size, learning_rate, optimizer, criterion, train_dataset, val_dataset=None, lr_scheduler=None, device=torch.device('cpu')):    
        self.basedir = basedir
        self.model = model

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.foldername = self._getFolderName()
        self.device = device

        self.TIME_NOW = time.time()

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
    
    def _getTrainLoader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def _getValLoader(self):
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def _createFolder(self):
        os.makedirs(os.path.join(self.basedir, self.foldername))
    
    def _saveSummaryModel(self):
        with open(os.path.join(self.basedir, self.foldername, 'summary.txt'), 'w+', encoding="utf-8") as f:
            f.write(str(summary(self.model)))

        with open(os.path.join(self.basedir, self.foldername, 'detail.txt'), 'w+', encoding="utf-8") as f:
            f.write(str(self.model))

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
    
    def _valid(self, epoch=0):
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
        self.logger.info(f'EPOCH: [{epoch:3d},{self.epochs:3d}]   | VAL-ACCURACY: {np.mean(val_acc):.4f}      | VAL-LOSS: {np.mean(val_loss):.4f}   --  {time.time()-self.TIME_NOW:.2f}s')
    
    def _trainEpoch(self, epoch=0):
        running_loss = 0
        total_loss = 0

        N1 = len(self.train_loader)

        self.model.train(True)
        for i, sample in enumerate(self.train_loader):
            x, y = sample

            x = x.to(self.device)
            y = y.to(self.device)

            ypred = self.model(x)

            loss = self.criterion(y, ypred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if i % 5 == 4:
                self.logger.info(f'EPOCH: [{epoch:3d},{self.epochs:3d}]   | MINI-BATCH: [{i + 1:4d},{N1:4d}]   | LOSS: {running_loss / 5:.4f}   --  {time.time()-self.TIME_NOW:.2f}s')
                running_loss = 0

        self.loss.append(total_loss / N1)

    def train(self, valid=True):
        if valid != (self.val_dataset is not None):
            raise Exception("Validation dataset not provided!")

        self._createFolder()
        self._initLog()
        self._getTrainLoader()
        if valid:
            self._getValLoader()
        self.accuracy = []
        self.loss = []

        self.TIME_NOW = time.time()

        for epoch in range(1, self.epochs+1):
            self.logger.info(f'START EPOCH - {epoch}')
            self._trainEpoch(epoch)
            if valid:
                self._valid(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        self._saveSummaryModel()
        self._saveConfig()
        if valid:
            self._saveAccuracy()
        self._saveLoss()
        self._saveModel()
        self._stopLog()


class TestEngine(object):
    """
    Engine to test model
    Example:
    $ tengine = TestEngine(basedir=PATH, model=model, dataset=dataset, device=device, model_name="001"))
    $ tengine.test()
    """
    def __init__(self, basedir, model, dataset, device, model_name: str=None):    
        self.basedir = basedir
        self.model = model
        self.dataset = dataset

        self.device = device
        self.foldername = self._getFolderName() if not model_name else model_name

    def _initLog(self):
        self.logger = logging.getLogger(self.foldername + "_log")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s :: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

        file_handler = logging.FileHandler(os.path.join(self.basedir, self.foldername, 'testing.log'))
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
    
    def _getTestLoader(self):
        self.test_loader = DataLoader(self.dataset, batch_size=len(self.dataset)//100)

    def _createFolder(self):
        dir = os.path.join(self.basedir, self.foldername)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    def _savePrediction(self):
        with open(os.path.join(self.basedir, self.foldername, "predictions.csv"), 'w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            writer.writerow(['ImageId', 'Label'])
            for i in range(len(self.y_hat)):
                writer.writerow([i+1, int(self.y_hat[i])])

    def test(self):
        self._createFolder()
        self._initLog()
        self._getTestLoader()
        self.TIME_NOW = time.time()

        self.y_hat = []
        for i, x in enumerate(self.test_loader):
            ypred = self.model(x.to(self.device))
            predict_class = torch.argmax(ypred.cpu(), dim=-1)
            self.y_hat = np.concatenate((self.y_hat, predict_class))

            self.logger.info(f'BATCH: [{i + 1:4d},{100:4d}]   --  {time.time()-self.TIME_NOW:.2f}s')

        self._savePrediction()
        self._stopLog()
