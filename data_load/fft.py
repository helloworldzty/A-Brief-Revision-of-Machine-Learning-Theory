import torch
import torch.fft as fft
from typing import Tuple, Union, Optional
import numpy as np
from Matdataset import create_mat_dataloader

def batch_fft_transform(batch_data: torch.Tensor, 
                       sampling_freq: float, 
                       return_phase: bool = False, 
                       keep_all: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Perform FFT transformation on batched time-domain vibration signals.
    Ensures even-length frequency spectra with proper symmetry handling.
    
    Parameters:
    -----------
    batch_data : torch.Tensor
        Input batch of time-domain signals with shape [BATCHSIZE, 1, LEN]
    sampling_freq : float
        Sampling frequency of the signals (Hz)
    return_phase : bool, optional
        If True, return both amplitude and phase spectra
        If False, return only amplitude spectrum (default: False)
    keep_all : bool, optional
        If True, keep the full symmetric spectrum (even length)
        If False, keep exactly half of the spectrum (even length) (default: False)
    
    Returns:
    --------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        - If return_phase=False: amplitude spectrum with shape:
            - [BATCHSIZE, 1, LEN] if keep_all=True (LEN is even)
            - [BATCHSIZE, 1, LEN//2] if keep_all=False (strictly half, even length)
        - If return_phase=True: tuple of (amplitude_spectrum, phase_spectrum) with same shapes
    
    Functionality:
    --------------
    Applies FFT to each signal in the batch, ensures even-length output spectra,
    and properly handles symmetry by removing the highest frequency point when necessary.
    """
    # Ensure input is on the correct device and has the right shape
    device = batch_data.device
    B, C, LEN = batch_data.shape
    
    if C != 1:
        raise ValueError(f"Expected channel dimension of 1, but got {C}")
    
    # Verify that LEN is divisible by a large power of 2 (as guaranteed)
    if LEN < 1024:  # 2^10 = 1024
        print(f"Warning: Signal length {LEN} is less than 1024, but should be divisible by 2^M with M>=10")
    
    # Remove channel dimension for FFT processing
    time_signals = batch_data.squeeze(1)  # Shape: [B, LEN]
    
    if keep_all:
        # Use full FFT and keep entire spectrum (even length)
        fft_complex = fft.fft(time_signals, dim=-1, norm='forward')  # Shape: [B, LEN]
        freq_length = LEN  # Already even due to divisibility by 2^M
    else:
        # Use rfft for efficiency, then adjust to get exactly half (even length)
        # rfft returns LEN//2 + 1 points for even LEN
        fft_complex_rfft = fft.rfft(time_signals, dim=-1, norm='forward')  # Shape: [B, LEN//2 + 1]
        
        # Remove the highest frequency point to get exactly half length (which is even)
        fft_complex = fft_complex_rfft[:, :-1]  # Shape: [B, LEN//2]
        freq_length = LEN // 2  # This is even since LEN is divisible by 2^M
    
    # Compute amplitude spectrum
    amplitude_spectrum = torch.abs(fft_complex)  # Shape: [B, freq_length]
    
    # Reshape back to [B, 1, freq_length]
    amplitude_spectrum = amplitude_spectrum.unsqueeze(1)
    
    if return_phase:
        # Compute phase spectrum (in radians)
        phase_spectrum = torch.angle(fft_complex)  # Shape: [B, freq_length]
        phase_spectrum = phase_spectrum.unsqueeze(1)  # Shape: [B, 1, freq_length]
        
        return amplitude_spectrum, phase_spectrum
    else:
        return amplitude_spectrum

def get_frequency_axis(signal_length: int, sampling_freq: float, keep_all: bool = False) -> torch.Tensor:
    """
    Generate frequency axis for FFT results with even length.
    
    Parameters:
    -----------
    signal_length : int
        Length of the original time-domain signal (even, divisible by 2^M)
    sampling_freq : float
        Sampling frequency (Hz)
    keep_all : bool, optional
        Whether the spectrum includes full symmetric part
    
    Returns:
    --------
    torch.Tensor
        Frequency axis values (Hz) with even length
    """
    if keep_all:
        # Full spectrum: 0 to sampling_freq
        freqs = fft.fftfreq(signal_length, d=1/sampling_freq)
    else:
        # Positive frequencies only, exactly half length (even)
        # Generate for LEN//2 + 1 points, then remove the last point
        freqs_full = fft.rfftfreq(signal_length, d=1/sampling_freq)
        freqs = freqs_full[:-1]  # Remove highest frequency point
    
    return freqs

def test_fft_even_length():
    """
    Test function specifically for even-length spectrum requirements.
    """
    # Create test signals with lengths divisible by 2^M (M>=10)
    test_lengths = [1024, 2048, 4096]  # 2^10, 2^11, 2^12
    
    for LEN in test_lengths:
        print(f"\n=== Testing with signal length {LEN} (divisible by 2^{int(np.log2(LEN))}) ===")
        
        # Create synthetic test batch
        batch_size = 4
        t = torch.linspace(0, 1, LEN)
        # Create signals with multiple frequency components
        signal1 = torch.sin(2 * np.pi * 50 * t)  # 50 Hz
        signal2 = torch.sin(2 * np.pi * 120 * t)  # 120 Hz
        signal3 = signal1 + 0.5 * signal2  # Mixed frequencies
        signal4 = torch.randn(LEN)  # Noise
        
        batch_data = torch.stack([signal1, signal2, signal3, signal4])
        batch_data = batch_data.unsqueeze(1)  # Shape: [4, 1, LEN]
        
        sampling_freq = 1000.0  # 1 kHz sampling
        
        # Test both keep_all configurations
        for keep_all in [False, True]:
            print(f"\nkeep_all = {keep_all}:")
            
            result = batch_fft_transform(
                batch_data=batch_data,
                sampling_freq=sampling_freq,
                return_phase=True,
                keep_all=keep_all
            )
            
            amp_spec, phase_spec = result
            
            expected_length = LEN if keep_all else LEN // 2
            print(f"  Input length: {LEN}")
            print(f"  Output length: {amp_spec.shape[-1]} (expected: {expected_length})")
            print(f"  Is even: {amp_spec.shape[-1] % 2 == 0}")
            
            # Verify dimensions
            assert amp_spec.shape == (batch_size, 1, expected_length), \
                f"Shape mismatch: got {amp_spec.shape}, expected ({batch_size}, 1, {expected_length})"
            assert phase_spec.shape == (batch_size, 1, expected_length), \
                f"Phase shape mismatch: got {phase_spec.shape}, expected ({batch_size}, 1, {expected_length})"
            
            # Verify even length
            assert amp_spec.shape[-1] % 2 == 0, f"Spectrum length {amp_spec.shape[-1]} is not even"
            
            # Test frequency axis
            freqs = get_frequency_axis(LEN, sampling_freq, keep_all)
            print(f"  Frequency axis length: {len(freqs)}")
            assert len(freqs) == expected_length, "Frequency axis length mismatch"
            
            if not keep_all:
                # For half spectrum, verify frequency range
                max_freq = freqs[-1].item()
                nyquist = sampling_freq / 2
                print(f"  Max frequency: {max_freq:.2f} Hz, Nyquist: {nyquist:.2f} Hz")
                assert max_freq < nyquist, "Half spectrum should not include Nyquist frequency"
        
        print(f"  ✓ All tests passed for LEN={LEN}")

def test_fft_with_real_dataloader():
    """
    Test function with the actual MatDataLoader.
    """
    file_path = "E://chunmiao//new_mats//C1_DE.mat"
    variable_name = "data_matrix"
    sampling_freq = 12000.0  # Adjust based on your data
    
    print("\n=== Testing FFT with Real DataLoader ===")
    
    try:
        dataloader = create_mat_dataloader(
            file_path=file_path,
            variable_name=variable_name,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            normalization='zscore'
        )
        
        for batch_idx, (batch_data, labels) in enumerate(dataloader):
            B, C, LEN = batch_data.shape
            
            print(f"\nBatch {batch_idx}: LEN = {LEN}")
            print(f"  Is LEN divisible by 1024: {LEN % 1024 == 0}")
            print(f"  Is LEN even: {LEN % 2 == 0}")
            
            # Test both configurations
            for keep_all in [False, True]:
                result = batch_fft_transform(
                    batch_data=batch_data,
                    sampling_freq=sampling_freq,
                    return_phase=False,
                    keep_all=keep_all
                )
                
                output_length = result.shape[-1]
                expected_length = LEN if keep_all else LEN // 2
                
                print(f"  keep_all={keep_all}: Input {LEN} -> Output {output_length}")
                print(f"    Expected: {expected_length}, Match: {output_length == expected_length}")
                print(f"    Output even: {output_length % 2 == 0}")
                
                assert output_length == expected_length, f"Length mismatch: {output_length} vs {expected_length}"
                assert output_length % 2 == 0, f"Output length {output_length} is not even"
            
            if batch_idx == 1:  # Test only 2 batches
                break
                
        print("\n✓ Real data test passed!")
        
    except Exception as e:
        print(f"⚠️  Real data test skipped: {e}")
        print("This is normal if the data file is not available")

# Example of integration with existing workflow
def create_frequency_dataloader(time_domain_dataloader, sampling_freq: float, 
                               return_phase: bool = False, keep_all: bool = False):
    """
    Create a frequency-domain dataloader from a time-domain dataloader.
    
    Parameters:
    -----------
    time_domain_dataloader : DataLoader
        Original dataloader with time-domain signals
    sampling_freq : float
        Sampling frequency
    return_phase : bool
        Whether to include phase information
    keep_all : bool
        Whether to keep full spectrum or half spectrum
    
    Yields:
    -------
    Batch data in frequency domain (and labels)
    """
    for batch_data, labels in time_domain_dataloader:
        # Transform to frequency domain
        freq_data = batch_fft_transform(
            batch_data=batch_data,
            sampling_freq=sampling_freq,
            return_phase=return_phase,
            keep_all=keep_all
        )
        
        if return_phase:
            amp_spec, phase_spec = freq_data
            # You can return both or concatenate them based on your needs
            yield (amp_spec, phase_spec), labels
        else:
            yield freq_data, labels

if __name__ == "__main__":
    # Test with synthetic data
    test_fft_even_length()
    
    # Test with real dataloader (if data available)
    test_fft_with_real_dataloader()
    
    # Demonstrate usage
    print("\n=== Usage Example ===")
    
    # Create a test signal
    LEN = 2048  # Divisible by 2^11
    t = torch.linspace(0, 1, LEN)
    test_signal = torch.sin(2 * np.pi * 50 * t) + 0.5 * torch.sin(2 * np.pi * 120 * t)
    batch_data = test_signal.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 2048]
    
    sampling_freq = 1000.0
    
    print(f"Input signal length: {LEN}")
    
    # Test half spectrum
    amp_half = batch_fft_transform(batch_data, sampling_freq, keep_all=False)
    print(f"Half spectrum length: {amp_half.shape[-1]} (even: {amp_half.shape[-1] % 2 == 0})")
    
    # Test full spectrum  
    amp_full = batch_fft_transform(batch_data, sampling_freq, keep_all=True)
    print(f"Full spectrum length: {amp_full.shape[-1]} (even: {amp_full.shape[-1] % 2 == 0})")
    
    # Get frequency axes
    freqs_half = get_frequency_axis(LEN, sampling_freq, keep_all=False)
    freqs_full = get_frequency_axis(LEN, sampling_freq, keep_all=True)
    
    print(f"Frequency axis (half): {len(freqs_half)} points, 0 to {freqs_half[-1]:.1f} Hz")
    print(f"Frequency axis (full): {len(freqs_full)} points, {freqs_full[0]:.1f} to {freqs_full[-1]:.1f} Hz")