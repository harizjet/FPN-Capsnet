import torch
import numpy as np
import cv2


def accuracy(y, ypred):
    """
    Get accuracy from probability
    """
    N = len(y)
    correct = 0
    predict_class = torch.argmax(ypred, dim=-1)
    for i in range(N):
        if predict_class[i] == y[i]:
            correct += 1
            
    return correct / N

def isValidModel(model, sample_X):
    """
    Check if the model is setup correctly
    """
    model.train(False)

    try:
        model(sample_X.to(model.device))
        return True
    except Exception as error:
        print(error)
        return False

def sharpen(img: np.array, kernel: np.array, strength: int=1):
    """
    Sharpening an images
    """
    return cv2.filter2D(img.astype(np.uint8), -1, kernel*strength)

def extractSIFT(img: np.array, thres: int=5):
    """
    Extract SIFT features
    """
    sift = cv2.SIFT_create(edgeThreshold=thres)
    key_point, feature = sift.detectAndCompute(img.astype('uint8'), None)
    return key_point, feature

def extractORB(img: np.array, thres: int=5):
    """
    Extract ORB features
    """
    orb = cv2.ORB_create(edgeThreshold=thres)
    key_point, feature = orb.detectAndCompute(img.astype('uint8'), None)
    return key_point, feature