import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from typing import Optional, Union, Literal, Dict

class MatDataLoader:
    '''
    Load data_files(.mat) WITH OPTIONAL NORMALIZATION
    
    Inputs:
        .mat: [data_num, 2^N+1]
        2^N: data_seq(float32)
        data[-1]:label(-0-num_classes-1)
        
    Outputs:
        dataloader.
        for each batch in the dataloader:
            data : [B,1,LEN],Len=2^N
            label : [B]
    
    Parameters:
    - file_path:str
    - variable_name: str
    - batch_size
    - shuffle:
    - num_workers
    - normalization: optional normalization method
    '''
    
    def __init__(self, file_path:str, variable_name:str, 
                 batch_size: int=32, shuffle:bool = True,
                 num_workers:int=4,
                 normalization: Optional[Literal['zscore', 'minmax']] = None,
                 eps: float = 1e-8):
        
        self.file_path = file_path
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalization = normalization
        self.eps = eps
        
        self.data, self.labels, self.sequence_length = self._load_mat_data()
        
        # Precompute global statistics if needed for z-score normalization
        self.global_stats = self._compute_global_stats() if normalization == 'zscore' else None
        
        self.dataset = self._create_dataset()
        
        self.dataloader = self._create_dataloader()
        
    def _load_mat_data(self):
        '''
        Processing .mat and extracting data&labels
         Return:
             data:[data_num,seq_len]
             labels: [data_num]
             seq_len : 2^N
        '''
        mat_data = sio.loadmat(self.file_path)
        if self.variable_name not in mat_data:
            raise ValueError(f"Variable: '{self.variable_name}' is not in the file")
        data_matrix = mat_data[self.variable_name]
        if data_matrix.ndim != 2:
            raise ValueError(f"2 dimentions are expected, but get{data_matrix.ndim} dimention(s) instead.")
        
        data_num, total_cols = data_matrix.shape
        
        sequence_length = total_cols - 1
        
        data = data_matrix[:, :-1]  
        labels = data_matrix[:, -1] 
        
        labels = labels.astype(np.int64)
        
        print(f"  - data_num: {data_num}")
        print(f"  - seq_len: {sequence_length}")
        print(f"  - label_range: {np.min(labels)} - {np.max(labels)}")
        print(f"  - data_dtype: {data.dtype}")
        print(f"  - normalization: {self.normalization}")
        
        return data, labels, sequence_length
    
    def _compute_global_stats(self) -> Dict[str, float]:
        """Compute global mean and std for z-score normalization"""
        mean = np.mean(self.data)
        std = np.std(self.data)
        print(f"  - global mean: {mean:.6f}, global std: {std:.6f}")
        return {'mean': mean, 'std': std}
    
    def _create_dataset(self):
        """Create PyTorch dataset"""
        return MatDataset(self.data, self.labels, self.normalization, self.global_stats, self.eps)
    
    def _create_dataloader(self):
        """Create PyTorch dataloader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def get_dataloader(self):
        """Access dataloader"""
        return self.dataloader
    
    def get_data_info(self):
        """Get data information"""
        return {
            'num_samples': len(self.data),
            'sequence_length': self.sequence_length,
            'num_classes': len(np.unique(self.labels)),
            'data_shape': f"[B, 1, {self.sequence_length}]",
            'label_shape': "[B]",
            'normalization': self.normalization
        }

class MatDataset(Dataset):
    """Custom dataset class with normalization support"""
    
    def __init__(self, data, labels, 
                 normalization: Optional[Literal['zscore', 'minmax']] = None,
                 global_stats: Optional[Dict[str, float]] = None,
                 eps: float = 1e-8):
        """
        Parameters:
        - data: raw data, shape [num_samples, sequence_length]
        - labels: labels, shape [num_samples]
        - normalization: normalization method
        - global_stats: precomputed statistics for z-score normalization
        - eps: small constant to avoid division by zero
        """
        self.data = torch.FloatTensor(data)  # Convert to float32 tensor
        self.labels = torch.LongTensor(labels)  # Convert to int64 tensor
        self.normalization = normalization
        self.global_stats = global_stats
        self.eps = eps
        
        # Apply normalization if specified
        if self.normalization is not None:
            self.data = self._normalize_data(self.data)
        
        # Reshape data to [num_samples, 1, sequence_length]
        self.data = self.data.unsqueeze(1)
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Apply normalization to the data"""
        if self.normalization == 'zscore':
            return self._zscore_normalize(data)
        elif self.normalization == 'minmax':
            return self._minmax_normalize(data)
        else:
            return data
    
    def _zscore_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Z-score normalization: (data - mean) / (std + eps)"""
        if self.global_stats is not None:
            # Use precomputed global statistics
            mean = self.global_stats['mean']
            std = self.global_stats['std']
            normalized_data = (data - mean) / (std + self.eps)
        else:
            # Compute statistics for each sample individually
            # Keep dimensions for broadcasting: [num_samples, 1] for mean and std
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
            normalized_data = (data - mean) / (std + self.eps)
        
        return normalized_data
    
    def _minmax_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Min-max normalization: scale to [0, 1]"""
        # Compute min and max for each sample individually
        # Keep dimensions for broadcasting: [num_samples, 1] for min and max
        min_vals = data.min(dim=1, keepdim=True)[0]
        max_vals = data.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # If max == min, set range to 1
        
        normalized_data = (data - min_vals) / range_vals
        return normalized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return:
        - data: shape [1, sequence_length]
        - label: shape [1]
        """
        return self.data[idx], self.labels[idx]


def create_mat_dataloader(file_path: str, variable_name: str, 
                         batch_size: int = 32, shuffle: bool = True,
                         num_workers: int = 0,
                         normalization: Optional[Literal['zscore', 'minmax']] = None,
                         eps: float = 1e-8) -> DataLoader:
    """
    Create MAT file data loader convenience function
    
    Parameters:
    - file_path: .mat file path
    - variable_name: variable name in .mat file
    - batch_size: batch size (default 32)
    - shuffle: whether to shuffle data (default True)
    - num_workers: number of data loading workers (default 0)
    - normalization: normalization method ('zscore' or 'minmax')
    - eps: small constant to avoid division by zero (default 1e-8)
    
    Return:
    - PyTorch DataLoader object
    """
    loader = MatDataLoader(
        file_path=file_path,
        variable_name=variable_name,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        normalization=normalization,
        eps=eps
    )
    return loader.get_dataloader()


# Usage examples
if __name__ == "__main__":
    # Example usage
    file_path = "E://chunmiao//new_mats//C1_DE.mat"
    variable_name = "data_matrix"
    
    # Method 1: No normalization
    print("=== No Normalization ===")
    mat_loader = MatDataLoader(
        file_path=file_path,
        variable_name=variable_name,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        normalization=None
    )
    
    dataloader = mat_loader.get_dataloader()
    info = mat_loader.get_data_info()
    print("Data info:", info)
    
    # Method 2: Z-score normalization with global statistics
    print("\n=== Z-score Normalization ===")
    mat_loader_zscore = MatDataLoader(
        file_path=file_path,
        variable_name=variable_name,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        normalization='zscore'
    )
    
    # Method 3: Min-max normalization
    print("\n=== Min-max Normalization ===")
    mat_loader_minmax = MatDataLoader(
        file_path=file_path,
        variable_name=variable_name,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        normalization='minmax'
    )
    
    # Test dataloaders
    for name, loader in [("No Norm", dataloader), 
                         ("Z-score", mat_loader_zscore.get_dataloader()),
                         ("Min-max", mat_loader_minmax.get_dataloader())]:
        print(f"\nTesting {name}:")
        for batch_idx, (data, labels) in enumerate(loader):
            print(f"Batch {batch_idx}:")
            print(f"  Data shape: {data.shape}")  # Should be [B, 1, LEN]
            print(f"  Data range: {data.min():.6f} to {data.max():.6f}")
            print(f"  Label shape: {labels.shape}")  # Should be [B]
            
            if batch_idx == 0:  # Only show first batch info
                break