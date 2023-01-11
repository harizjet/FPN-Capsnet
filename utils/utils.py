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

def isValidModel(model, sample_X):
    """
    Check if the model is setup correctly
    """
    model.train(False)

    try:
        model(sample_X.to(model.device))
        return True
    except Exception as e:
        print(e)
        return False

def sharpen():
    pass
