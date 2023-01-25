import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools


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

def plot_confusion_matrix(cm, target_names, figsize=(13, 11), title='Confusion matrix', cmap=None, normalize=True):
    """
    Visualized confusion matrix
    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()