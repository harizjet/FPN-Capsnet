import torch


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

def sharpen():
    pass
