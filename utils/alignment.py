#--coding utf-8--
'''
Created at 8:14 Sept/23/2025
Some basic alignment losses utilized in UDA
'''

import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    '''
    Maximum Mean Discrepancy Loss
    Initialize:
        kernel_type: 'rbf', 'linear', 'poly'
        kernel_mul (float): Multiplier for RBF kernel bandwidth
        kernel_num (int): Number of kernels for multi-kernel MMD
        fix_sigma (float): Fixed sigma for RBF kernel. If None, use median heuristic
    
    Input:
        source_feat:shape:[B,feat_dim]
        target_feat:shape:[B,feat_dim]
        
    Output:
    '''
    
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss,self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        
    def guassian_kernel(self, source_feat, target_feat):
        """Calculate Gaussian kernel Matrix with improved implementation"""
        n_source = source_feat.size(0)
        n_target = target_feat.size(0)
        n_samples = n_source + n_target
        
        # More efficient way to compute pairwise distances
        X = torch.cat([source_feat, target_feat], dim=0)
        
        # Compute squared Euclidean distances using matrix operations
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        dist_sq = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        
        # Ensure distances are non-negative (numerical stability)
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        # Use median heuristic for bandwidth if not fixed
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            # Use median of pairwise distances (more robust)
            flat_dist_sq = dist_sq.flatten()
            # Exclude diagonal zeros
            non_zero_dist = flat_dist_sq[flat_dist_sq > 1e-8]
            if len(non_zero_dist) > 0:
                bandwidth = torch.median(non_zero_dist).sqrt()
            else:
                bandwidth = torch.tensor(1.0)
            
            # Avoid zero bandwidth
            if bandwidth < 1e-8:
                bandwidth = torch.tensor(1.0)
        
        # Multiple kernel bandwidths for multi-kernel MMD
        bandwidth_base = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth_base * (self.kernel_mul ** i) 
                         for i in range(self.kernel_num)]
        
        kernel_val = torch.zeros_like(dist_sq)
        for bw in bandwidth_list:
            kernel_val += torch.exp(-dist_sq / (2 * bw**2 + 1e-8))
            
        return kernel_val
    
    def linear_kernel(self, source, target):
        """Calculate linear kernel matrix"""
        total = torch.cat([source, target], dim=0)
        return torch.mm(total, total.t())

    def polynomial_kernel(self, source, target):
        """Calculate polynomial kernel matrix"""
        total = torch.cat([source, target], dim=0)
        gamma = 1.0 / total.size(1)
        degree = 3
        coef0 = 1
        K_linear = torch.mm(total, total.t())
        return (gamma * K_linear + coef0) ** degree
    
    def forward(self, source, target):
        """
        Calculate MMD loss between source and target features
    
        Args:
            source (Tensor): Source domain features [B, feat_dim]
            target (Tensor): Target domain features [B, feat_dim]
            
        Returns:
            Tensor: MMD loss value (non-negative)
        """
        if source.size(0) != target.size(0):
            raise ValueError("Source and target must have the same batch size")
        
        batch_size = source.size(0)
    
        # Calculate kernel matrix based on kernel type
        if self.kernel_type == 'linear':
            kernel_matrix = self.linear_kernel(source, target)
        elif self.kernel_type == 'poly':
            kernel_matrix = self.polynomial_kernel(source, target)
        else:  # 'rbf' as default
            kernel_matrix = self.guassian_kernel(source, target)
    
        # Extract sub-matrices for MMD calculation
        XX = kernel_matrix[:batch_size, :batch_size]
        YY = kernel_matrix[batch_size:, batch_size:]
        XY = kernel_matrix[:batch_size, batch_size:]
    
        # MMD formula: E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        # Use unbiased estimator (excluding diagonal)
        loss = (XX.sum() - XX.trace()) / (batch_size * (batch_size - 1)) + \
               (YY.sum() - YY.trace()) / (batch_size * (batch_size - 1)) - \
               2 * XY.sum() / (batch_size * batch_size)
        
        # Ensure non-negative loss (theoretical property of MMD)
        loss = torch.clamp(loss, min=0.0)
    
        return loss

class CORALLoss(nn.Module):
    '''
    CORAL (Correlation Alignment) Loss for domain adaptation
    
    Paper: Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016)
    
    Input:
        source_feat: shape [B, feat_dim]
        target_feat: shape [B, feat_dim]
        
    Output:
        CORAL loss value (scalar tensor)
    '''
    
    def __init__(self):
        super(CORALLoss, self).__init__()
        
    def compute_covariance(self, features):
        """
        Compute covariance matrix of features
        
        Args:
            features: Tensor of shape [n, d] where n is batch size, d is feature dimension
            
        Returns:
            covariance matrix of shape [d, d]
        """
        n = features.size(0)  # batch size
        
        # For small batch sizes, use biased estimator to avoid division by zero
        if n <= 1:
            # If batch size is 1 or 0, return zero covariance
            return torch.zeros(features.size(1), features.size(1), 
                             device=features.device, dtype=features.dtype)
        
        # Subtract mean
        features_mean = features.mean(dim=0, keepdim=True)
        features_centered = features - features_mean
        
        # Compute covariance matrix
        covariance = torch.mm(features_centered.t(), features_centered) / (n - 1)
        
        return covariance
    
    def forward(self, source, target):
        """
        Calculate CORAL loss between source and target features
        
        Args:
            source (Tensor): Source domain features [B, feat_dim]
            target (Tensor): Target domain features [B, feat_dim]
            
        Returns:
            Tensor: CORAL loss value (scalar)
        """
        if source.size(1) != target.size(1):
            raise ValueError("Source and target must have the same feature dimension")
            
        d = source.size(1)  # feature dimension
        
        # Handle edge case where feature dimension is 0
        if d == 0:
            return torch.tensor(0.0, device=source.device)
        
        # Compute covariance matrices
        cov_source = self.compute_covariance(source)
        cov_target = self.compute_covariance(target)
        
        # Compute CORAL loss: 1/(4d^2) * ||C_S - C_T||_F^2
        diff = cov_source - cov_target
        loss = torch.norm(diff, p='fro') ** 2  # Squared Frobenius norm
        loss = loss / (4 * d * d)  # Normalize by 4d^2
        
        # Ensure non-negative loss
        loss = torch.clamp(loss, min=0.0)
        
        return loss


