import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal

class ColoredNoiseGenerator:
    '''
    Generating colored Noise.
    '''
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._setup_filters()
        
    def _setup_filters(self):
        self.pink_b = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004408786], 
                                  dtype=torch.float32, device=self.device)
        self.pink_a = torch.tensor([1, -2.494956002, 2.017265875, -0.522189400], 
                                  dtype=torch.float32, device=self.device)
       
        self.brown_b = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        self.brown_a = torch.tensor([1.0, -1.0], dtype=torch.float32, device=self.device)

        self.blue_b = torch.tensor([1.0, -1.0], dtype=torch.float32, device=self.device)
        self.blue_a = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        self.violet_b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32, device=self.device)
        self.violet_a = torch.tensor([1.0], dtype=torch.float32, device=self.device)

noise_gen = ColoredNoiseGenerator()

def torch_lfilter(b, a, x):
    '''
    Apply a digital filter to a signal using PyTorch tensors.
    
    Parameters
    ----------
    b : torch.Tensor
        Numerator coefficients [b0, b1, ..., bM]
    a : torch.Tensor
        Denominator coefficients [a0, a1, ..., aN] 
    x : torch.Tensor
        Input signal with shape [B, 1, L]

    Returns
    -------
    y : torch.Tensor
        Filtered signal with shape [B, 1, L]
    '''

    batch_size, channels, length = x.shape
    
    b = b.to(x.device).to(x.dtype)
    a = a.to(x.device).to(x.dtype)
    
    y = torch.zeros_like(x)
    M = len(b) - 1
    N = len(a) - 1
    
    for n in range(length):

        y_current = b[0] * x[:, :, n]
        
        for i in range(1, M + 1):
            if n - i >= 0:
                y_current += b[i] * x[:, :, n - i]
        
        for i in range(1, N + 1):
            if n - i >= 0:
                y_current -= a[i] * y[:, :, n - i]
        
        y_current = y_current / a[0]  
        y[:, :, n] = y_current
    
    return y

def add_white_noise(data, snr_db=20):
    '''
    Add white noise to input signal.
    
    Parameters
    ----------
    data : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.

    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    '''
    device = data.device
    dtype = data.dtype
        
    white_noise = torch.randn_like(data)
        
    signal_power = torch.mean(data**2, dim=2, keepdim=True)  # [B, 1, 1]
    snr_linear = 10**(snr_db / 10)
    required_noise_power = signal_power / snr_linear
        
    current_noise_power = torch.mean(white_noise**2, dim=2, keepdim=True)
    scale_factor = torch.sqrt(required_noise_power / current_noise_power)
    scaled_noise = white_noise * scale_factor
        
    noisy_batch = data + scaled_noise
        
    return noisy_batch
    
    
def add_pink_noise(batch, snr_db=20):
    """
    Add pink noise (1/f noise) to input signal.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.
        
    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    white_noise = torch.randn(B, C, L, device=device, dtype=dtype)
    
    pink_noise = torch_lfilter(noise_gen.pink_b, noise_gen.pink_a, white_noise)
    
    pink_noise = pink_noise - torch.mean(pink_noise, dim=2, keepdim=True)
    
    signal_power = torch.mean(batch**2, dim=2, keepdim=True)
    snr_linear = 10**(snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    current_noise_power = torch.mean(pink_noise**2, dim=2, keepdim=True)
    scale_factor = torch.sqrt(required_noise_power / current_noise_power)
    scaled_noise = pink_noise * scale_factor
    
    noisy_batch = batch + scaled_noise
    
    return noisy_batch

def add_brown_noise(batch, snr_db=20):
    """
    Add brown noise (1/f² noise) to input signal.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.
        
    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    white_noise = torch.randn(B, C, L, device=device, dtype=dtype)
    
    brown_noise = torch.cumsum(white_noise, dim=2)
    
    brown_noise = brown_noise - torch.mean(brown_noise, dim=2, keepdim=True)
    
    signal_power = torch.mean(batch**2, dim=2, keepdim=True)
    snr_linear = 10**(snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    current_noise_power = torch.mean(brown_noise**2, dim=2, keepdim=True)
    scale_factor = torch.sqrt(required_noise_power / current_noise_power)
    scaled_noise = brown_noise * scale_factor
    
    noisy_batch = batch + scaled_noise
    
    return noisy_batch

def add_blue_noise(batch, snr_db=20):
    """
    Add blue noise (f noise) to input signal.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.
        
    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    white_noise = torch.randn(B, C, L, device=device, dtype=dtype)
    
    blue_noise = torch.diff(white_noise, n=1, dim=2)
    blue_noise = F.pad(blue_noise, (1, 0), mode='replicate')  
    
    signal_power = torch.mean(batch**2, dim=2, keepdim=True)
    snr_linear = 10**(snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    current_noise_power = torch.mean(blue_noise**2, dim=2, keepdim=True)
    scale_factor = torch.sqrt(required_noise_power / current_noise_power)
    scaled_noise = blue_noise * scale_factor
    
    noisy_batch = batch + scaled_noise
    
    return noisy_batch

def add_violet_noise(batch, snr_db=20):
    """
    Add violet noise (f² noise) to input signal.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.
        
    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    white_noise = torch.randn(B, C, L, device=device, dtype=dtype)
    
    violet_noise = torch.diff(white_noise, n=2, dim=2)
    violet_noise = F.pad(violet_noise, (2, 0), mode='replicate')  
    
    signal_power = torch.mean(batch**2, dim=2, keepdim=True)
    snr_linear = 10**(snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    current_noise_power = torch.mean(violet_noise**2, dim=2, keepdim=True)
    scale_factor = torch.sqrt(required_noise_power / current_noise_power)
    scaled_noise = violet_noise * scale_factor
    
    noisy_batch = batch + scaled_noise
    
    return noisy_batch

NOISE_FUNCTIONS = {
    'white': add_white_noise,
    'pink': add_pink_noise,
    'brown': add_brown_noise,
    'blue': add_blue_noise,
    'violet': add_violet_noise
}

def add_colored_noise(batch, noise_type='white', snr_db=20):
    """
    Add colored noise to input signal.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    noise_type : str, optional
        Type of noise ('white', 'pink', 'brown', 'blue', 'violet'). Default is 'white'.
    snr_db : float, optional
        Signal-to-noise ratio in dB. Default is 20.
        
    Returns
    -------
    torch.Tensor
        Noisy signal with shape [B, 1, L]
    """
    if noise_type not in NOISE_FUNCTIONS:
        raise ValueError(f"Unsupported noise type: {noise_type}. Available: {list(NOISE_FUNCTIONS.keys())}")
    
    return NOISE_FUNCTIONS[noise_type](batch, snr_db)

def add_random_noise(batch, snr_range=(10, 30)):
    """
    Add random colored noise with random SNR within specified range.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    snr_range : tuple, optional
        SNR range (min, max) in dB. Default is (10, 30).
        
    Returns
    -------
    tuple
        (noisy_batch, noise_info) where:
        - noisy_batch: Noisy signal with shape [B, 1, L]
        - noise_info: Dictionary containing noise type and SNR used
    """
    noise_types = list(NOISE_FUNCTIONS.keys())
    selected_noise = np.random.choice(noise_types)
    
    min_snr, max_snr = snr_range
    snr_db = np.random.uniform(min_snr, max_snr)
    
    noisy_batch = add_colored_noise(batch, selected_noise, snr_db)
    
    noise_info = {
        'type': selected_noise,
        'snr_db': snr_db
    }
    
    return noisy_batch, noise_info


def time_shift(batch, shift_ratio=None):
    """
    Apply time shift augmentation by swapping signal segments at random breakpoint.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    shift_ratio : float, optional
        Breakpoint position ratio. If None, randomly selected.
        
    Returns
    -------
    torch.Tensor
        Time-shifted signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    shifted_batch = batch.clone()
    
    for i in range(B):
        if shift_ratio is None:
            shift_point = torch.randint(1, L-1, (1,), device=device).item()
        else:
            shift_point = int(L * shift_ratio)
            shift_point = max(1, min(shift_point, L-1))
        
        signal = batch[i, :, :]
        shifted_batch[i, :, :] = torch.cat([signal[:, shift_point:], signal[:, :shift_point]], dim=1)
    
    return shifted_batch


def random_masking(batch, mask_ratio=0.1):
    """
    Apply random masking augmentation by setting random sampling points to zero.
    
    Parameters
    ----------
    batch : torch.Tensor
        Input signal with shape [B, 1, L]
    mask_ratio : float, optional
        Ratio of sampling points to mask (0-1). Default is 0.1.
        
    Returns
    -------
    torch.Tensor
        Masked signal with shape [B, 1, L]
    """
    device = batch.device
    dtype = batch.dtype
    B, C, L = batch.shape
    
    masked_batch = batch.clone()
    
    for i in range(B):
        num_mask = int(L * mask_ratio)
        
        if num_mask > 0:
            mask_indices = torch.randperm(L, device=device)[:num_mask]
            masked_batch[i, :, mask_indices] = 0
    
    return masked_batch


class DataAugmentation:
    """
    Data augmentation class integrating all augmentation methods.
    """
    
    def __init__(self, augmentation_config, num_augmentations=1):
        """
        Initialize data augmentation class.
        
        Parameters
        ----------
        augmentation_config : dict
            Dictionary with format {"method_name": [params, probability], ...}
        num_augmentations : int, optional
            Number of augmentation methods to randomly select. Default is 1.
        """
        self.augmentation_config = augmentation_config
        self.num_augmentations = num_augmentations
        
        self.augmentation_functions = {
            'white_noise': lambda batch, snr_db: add_colored_noise(batch, 'white', snr_db),
            'pink_noise': lambda batch, snr_db: add_colored_noise(batch, 'pink', snr_db),
            'brown_noise': lambda batch, snr_db: add_colored_noise(batch, 'brown', snr_db),
            'blue_noise': lambda batch, snr_db: add_colored_noise(batch, 'blue', snr_db),
            'violet_noise': lambda batch, snr_db: add_colored_noise(batch, 'violet', snr_db),
            'time_shift': time_shift,
            'random_masking': random_masking
        }
    
    def __call__(self, batch):
        """
        Apply randomly selected augmentation methods to input data.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input signal with shape [B, 1, L]
            
        Returns
        -------
        tuple
            (augmented_batch, applied_augmentations) where:
            - augmented_batch: Augmented signal with shape [B, 1, L]
            - applied_augmentations: List of applied augmentation method information
        """
        method_names = list(self.augmentation_config.keys())
        probabilities = [self.augmentation_config[name][1] for name in method_names]
        
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(method_names)] * len(method_names)
        
        selected_methods = np.random.choice(
            method_names, 
            size=min(self.num_augmentations, len(method_names)), 
            replace=False, 
            p=probabilities
        )
        
        augmented_batch = batch.clone()
        applied_augmentations = []
        
        for method_name in selected_methods:
            if method_name in self.augmentation_functions:
                params = self.augmentation_config[method_name][0]
                
                if method_name.endswith('_noise'):
                    augmented_batch = self.augmentation_functions[method_name](augmented_batch, params)
                elif method_name == 'time_shift':
                    augmented_batch = self.augmentation_functions[method_name](augmented_batch, params)
                elif method_name == 'random_masking':
                    augmented_batch = self.augmentation_functions[method_name](augmented_batch, params)
                
                applied_augmentations.append({
                    'method': method_name,
                    'params': params
                })
        
        return augmented_batch, applied_augmentations
    
    def get_available_methods(self):
        """
        Get all available augmentation method names.
        
        Returns
        -------
        list
            List of available augmentation method names
        """
        return list(self.augmentation_functions.keys())

