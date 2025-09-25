from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainDiscriminator(nn.Sequential):
    '''
    Domain-Adversarial Training of Neural Networks (ICML 2015)
    
    Distinguishing whether the input features come from the source domain or the target domain
    source_domain = 1, target_domain = 0
    
    Args:
        in_feature: dimention of the input features(int)
        hidden_size : dimention of the hidden layers(int)
        batch_norm : whether use: class:torch.nn.Batchnorm1d
            False: Use torch.nn.Dropout
            Default: True
    
    Input Shape:
        Inputs: [B, in_feature]
        Outputs: [B, 1] (Sigmoid=True) or [B, 2] (sigmoid=False)
    '''
    def __init__(self, in_feature:int , hidden_size:int , batch_norm=True, sigmoid = True):
        '''
        Args:
            in_feature (int): dimension of input features
            hidden_size (int): dimension of hidden layers
            batch_norm (bool): whether to use batch normalization
            sigmoid (bool): whether to use sigmoid output (binary) or linear output (2-class)
        '''
        if sigmoid:
            final_layer = nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)

        if batch_norm:
            super(DomainDiscriminator,self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
                )
            
        else:
            super(DomainDiscriminator,self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                final_layer
                )
            
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
            x (torch.Tensor): input features with shape [batch_size, in_feature]
        
        Outputs:
            torch.Tensor: domain predictions with shape [batch_size, 1] (sigmoid=True) 
                         or [batch_size, 2] (sigmoid=False)
        '''
        return super().forward(x)
            
class DomainAdversarialLoss(nn.Module):
    '''
    To calculate Domain Adversarial Loss
    
    Args: 
        sigmoid (bool): True: discriminator outputs [B, 1] with sigmoid
                       False: discriminator outputs [B, 2] with linear layer
    '''     
    def __init__(self, sigmoid=True):
        '''
        Args:
            sigmoid (bool): whether the discriminator uses sigmoid output
        '''
        super(DomainAdversarialLoss,self).__init__()
        self.sigmoid = sigmoid
        
    def forward(self, domain_pred, domain_label = 'source'):
        '''
        Inputs:
            domain_pred (torch.Tensor): domain discriminator predictions with shape
                                       [batch_size, 1] (sigmoid=True) or [batch_size, 2] (sigmoid=False)
            domain_label (str): domain label, either 'source' or 'target'
        
        Outputs:
            torch.Tensor: scalar loss value with shape []
        '''
        assert domain_label in ['source', 'target']
        if self.sigmoid:
            if domain_label == 'source':
                return F.binary_cross_entropy(domain_pred, torch.ones_like(domain_pred).to(domain_pred.device))
            else:
                return F.binary_cross_entropy(domain_pred, torch.zeros_like(domain_pred).to(domain_pred.device))
        else:
            if domain_label == 'source':
                targets = torch.ones(domain_pred.size(0), dtype=torch.long).to(domain_pred.device)
            else:
                targets = torch.zeros(domain_pred.size(0), dtype=torch.long).to(domain_pred.device)
            return F.cross_entropy(domain_pred, targets)