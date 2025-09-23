import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

def Entropy(input_, reduction):
    '''
    Input:
        input_ : shape[B, num_classes]
        reduction: sum/mean/ori
    Output 
        if reduction==sum or reduction==mean:
            a number(float)
        else:
            entropy:shape[B]
    '''
    eps = 1e-8
    entropy = -input_ * torch.log(input_ + eps)

    if reduction == 'sum':
        return torch.sum(entropy).item()
    elif reduction == 'mean':
        return torch.mean(entropy).item()
    else:
        return entropy

class CrossEntropyLossSmooth(nn.Module):
    '''
        CrossEntropy Loss
    Equation: y = - log(y_correct)
    Args:
        __init__:
            num_classes: (int) number of classes
            epsilon(float): weight.
    '''
    def __init__(self, num_classes, epsilon=1e-8, use_gpu = True, reduction= True):
        super(CrossEntropyLossSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        '''
        Args:
            inputs:prediction matrix before softmax with shape[B, num_classes]
            targets:ground truth labels with shape[batch_size]

        Outputs:
            if self.reduction == True: return CrossEntropy(float)
            else: return shape[Batch_size]
        '''
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        


class Divergence(nn.Module):
    '''
    Divergence Loss for two probability distributions
    Args:
        __init__:
            num_classes: (int) number of classes
            divergence_type: (str) type of divergence: 'kl' or 'js'
            use_gpu: (bool) whether to use GPU
            reduction: (bool) whether to reduce the loss
            softmax: (bool) whether to apply softmax to inputs
    '''
    def __init__(self, num_classes, divergence_type='kl', use_gpu=True, reduction=True, softmax=True):
        super(Divergence, self).__init__()
        self.num_classes = num_classes
        self.divergence_type = divergence_type
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.softmax = softmax

    def forward(self, p, q):
        '''
        Args:
            p: first probability distribution with shape [B, C]
            q: second probability distribution with shape [B, C]
            
        Outputs:
            if self.reduction == True: return divergence loss (float)
            else: return shape [Batch_size]
        '''
        # Apply softmax if needed
        if self.softmax:
            p = F.softmax(p, dim=1)
            q = F.softmax(q, dim=1)
        
        # Calculate divergence based on type
        if self.divergence_type == 'kl':
            loss = self._kl_divergence(p, q)
        elif self.divergence_type == 'js':
            loss = self._js_divergence(p, q)
        else:
            raise ValueError(f"Unsupported divergence type: {self.divergence_type}")
        
        if self.reduction:
            return loss.mean()
        else:
            return loss

    def _kl_divergence(self, p, q):
        """KL divergence: D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))"""
        return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=1)

    def _js_divergence(self, p, q):
        """JS divergence: symmetric KL divergence"""
        m = 0.5 * (p + q)
        return 0.5 * (self._kl_divergence(p, m) + self._kl_divergence(q, m))

