import torch

class function(object):
    def __init__(self, fx: object):
        self.fx = fx
    
    def __call__(self, data):
        sample = self.fx(data[0])
        output = []
        for i in range(len(sample)):
            output.append(sample[i].unsqueeze(0))
        for d in data[1:]:
            res = self.fx(d)
            for i in range(len(res)):
                output[i] = torch.cat((output[i], res[i].unsqueeze(0)))
        if len(output) == 1:
            return output[0]
        return output