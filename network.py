# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:02 2025

@author: ZhangTianyu Beihang University
"""

import torch
import torch.nn as nn

class Layer5CNN(nn.Module):
    '''
    Inputs: data: shape=[batch_size>1, data_len]
            feat_dim (int, default=512)
    Outputs: feat: shape=[batch_size>1, feat_dim]
    '''
    def __init__(self,feat_dim=512):
        super(Layer5CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #Input Layer
            nn.Conv1d(in_channels=1, out_channels = 64, kernel_size=128, stride=12),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            
            #Hidden Layer1
            nn.Conv1d(in_channels=64, out_channels=32,kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            
            #Hidden Layer2
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            
            #Hidden Layer3
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            
            #Linear Layer
            nn.Flatten(),
            nn.Linear(592, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )
        
    def forward(self, x):
        return self.conv_layers(x)
    
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18_1D(nn.Module):
    def __init__(self, feat_dim, input_channels=1):
        super(ResNet18_1D, self).__init__()
        self.feat_dim = feat_dim
        
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, feat_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''

        Parameters
        ----------
        x : [B,DATA_LEN]

        Returns
        -------
        feat : [B,FEAT_DIM]

        '''

        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x
    
class Classifier(nn.Module):
    '''
    Inputs:
        __init__: feat_dim, num_classes, dropout_rate=0.5
        forward: [B,feat_dim]
    Outputs:
        [B,num_classes]
    '''
    def __init__(self, feat_dim, num_classes, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(

            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(feat_dim // 2, num_classes)

        )
    def forward(self,x):
        '''
        Inputs:
            x:[B,feat_dim]
        Outputs:
            logits:[B,num_classes]
        '''
        return self.classifier(x)
    
