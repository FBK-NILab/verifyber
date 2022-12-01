import torch.nn as nn
import torch.nn.functional as F 

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target.long())
