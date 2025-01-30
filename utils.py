import mne
import pandas as pd
import os
from typing import Union, Tuple
import numpy as np
import os
import mne
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy import stats


# most of this code is inspired from : https://github.com/mne-tools/mne-python/issues/5684
def PSG_to_csv(path):
    """
    Convertion of Polysomnography .edf files to .csv
    """

    # design path // data id can be changed
    psg_file = os.path.join(path, "SC4182E0-PSG.edf")

    # load .edf data // data id can be changed
    raw_psg = mne.io.read_raw_edf(psg_file, preload=True)

    # extract data and channel names
    data, times = raw_psg.get_data(return_times=True)  # raw data and timestamps
    channel_names = raw_psg.ch_names  # channel names

    # create a DataFrame with the data
    df = pd.DataFrame(data.T, columns=channel_names)
    df['Time (s)'] = times  # Add the timestamps as a column

    # saving to .csv
    psg_csv_file = os.path.join(path,"SC4182E0-PSG.edf")
    df.to_csv(psg_csv_file, index=False)
    print(f"EDF file has been converted and saved as: {psg_csv_file}")


def Hypnogram_to_csv(path): 
    """
    Convertion of corresponding Hypnogram .edf files to .csv
    """

    # path to the Hypnogram EDF file
    hypnogram_file = os.path.join(path,"SC4182EC-Hypnogram.edf")

    # load the annotations from the Hypnogram file
    annotations = mne.read_annotations(hypnogram_file)
    print("Annotations content:")
    print(annotations)

    # convert annotations to a DataFrame
    annotations_df = pd.DataFrame({
        'Onset (s)': annotations.onset,       # Start time of the annotation
        'Duration (s)': annotations.duration, # Duration of the annotation
        'Description': annotations.description # Description (e.g., sleep stages)
    })

    # save the annotations df to .csv fiels
    hyp_csv_file = os.path.join(path,"SC4182EC-Hypnogram.csv")
    annotations_df.to_csv(hyp_csv_file, index=False)
    print(f"Hypnogram annotations have been converted and saved as: {hyp_csv_file}")


#############################################################################################

def load_and_normalize_eeg_data(psg_file, hypnogram_file):
    """
    Load and normalize EEG data and corresponding hypnogram for a single patient.

    Parameters
    ----------
    psg_file : str
        Path to the PSG file (e.g., 'SC4182E0-PSG.edf').
    hypnogram_file : str
        Path to the hypnogram file (e.g., 'SC4182EC-Hypnogram.edf').

    Returns
    -------
    data_dict : dict
        Dictionary containing:
            - 'psg_name': str
            - 'hypnogram_name': str
            - 'subject_id': int
            - 'night_number': int
            - 'annotation_id': int
            - 'normalized_signal_data': array, shape (n_epochs, 2, n_time_steps)
            - 'epoch_onset_times': array, shape (n_epochs,)
            - 'labels': array, shape (n_epochs,)
    """
    # mappings for annotations
    annotation_desc_2_event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4,
        'Sleep stage R': 5
    }

    # unifie stages 3 and 4 under a single label (according to AASM guidelines specified in the paper)
    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4,
        #'Sleep_stage 4': 3,
        'Sleep stage R': 5
    }

    # Read the raw PSG data and the corresponding annotations
    raw_file = mne.io.read_raw_edf(psg_file, preload=True)
    annotations = mne.read_annotations(hypnogram_file)

    # Set annotations in the raw object
    raw_file.set_annotations(annotations, emit_warning=False)

    # Crop edges to remove leading and trailing times with no annotation (see paper)
    annotations.crop(
        annotations[1]['onset'] - 30 * 60,
        annotations[-2]['onset'] + 30 * 60
    )
    raw_file.set_annotations(annotations, emit_warning=False)

    # Extract 30-second events from the annotations
    events, _ = mne.events_from_annotations(
        raw_file, event_id=annotation_desc_2_event_id, chunk_duration=30.
    )

    # Epoch parameters
    sfreq = raw_file.info['sfreq']
    epoch_tmax = 30. - 1. / sfreq

    epochs = mne.Epochs(
        raw=raw_file,
        events=events,
        event_id=event_id,
        tmin=0.,
        tmax=epoch_tmax,
        baseline=None
    )

    # Extract signal data for the desired channels
    signal_data = epochs.get_data(picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])

    # Normalize each channel across all epochs and time points
    normalized_signal_data = signal_data.copy()
    for channel in range(normalized_signal_data.shape[1]):
        channel_data = normalized_signal_data[:, channel, :]
        channel_mean = channel_data.reshape(-1).mean()
        channel_std = channel_data.reshape(-1).std()
        normalized_signal_data[:, channel, :] = (channel_data - channel_mean) / channel_std

    # Build the output dictionary
    data_dict = {
        'psg_name': os.path.basename(psg_file),
        'hypnogram_name': os.path.basename(hypnogram_file),
        'subject_id': int(os.path.basename(psg_file)[3:5]),
        'night_number': int(os.path.basename(psg_file)[5]),
        'annotation_id': int(os.path.basename(psg_file)[7]),
        'normalized_signal_data': normalized_signal_data,
        'epoch_onset_times': epochs.events[:, 0],
        'labels': epochs.events[:, 2]
    }

    return data_dict


#############################################################################################


    
        


#############################################################################################
#                                     FEATURES
#############################################################################################

def compute_fractal_dimension(time_series: np.ndarray, 
                            min_box_size: int = 2,
                            max_box_size: int = None) -> float:
    """
    Compute fractal dimension of a time series using box-counting method.
    
    Args:
        time_series: Input signal
        min_box_size: Minimum box size (default: 2)
        max_box_size: Maximum box size (default: len(time_series)/10)
    
    Returns:
        float: Fractal dimension (slope of ln(L) vs ln(S(L)/L))
    """
    N = len(time_series)
    # if not specified take: max_box_size = N // 10 
    if max_box_size is None:
        max_box_size = N // 10  

    # Generate range of box sizes by log spacing them
    box_sizes = np.logspace(np.log10(min_box_size), 
                           np.log10(max_box_size),
                           num=20,
                           dtype=int)
    
    # Remove duplicates and sort
    box_sizes = np.unique(box_sizes)
    
    # Store results for each box size
    counts = []
    
    for box_size in box_sizes:
        # Number of boxes
        n_boxes = N // box_size
        
        # Initialize count for this box size
        box_count = 0
        
        for i in range(n_boxes):
            # Extract segment for current box
            start_idx = i * box_size
            end_idx = min((i + 1) * box_size, N)
            segment = time_series[start_idx:end_idx]
            
            # Compute max-min difference as per paper
            # S(n∆t) = Σ|max(∆xi) - min(∆xi)|
            box_count += np.abs(np.max(segment) - np.min(segment))
            
        counts.append(box_count)
    
    
    # Prepare data for linear regression
    counts = np.array(counts)
    box_sizes = np.array(box_sizes)
    # Using ln(L) vs ln(S(L)/L) as per paper
    x = np.log(box_sizes)
    y = np.log(counts / box_sizes)    
    
    # Linear fit to get the slope (FD)
    coeffs = np.polyfit(x, y, 1)
    fractal_dim = coeffs[0]  # Slope gives fractal dimension
    
    return -fractal_dim

def compute_DFA(signal: np.ndarray, scales: np.ndarray = None) -> Tuple[float, float, float]:
    """
    Compute Detrended Fluctuation Analysis as described in the paper.
    
    Args:
        signal: Input time series
        scales: Array of window sizes to use for analysis
                If None, will generate logarithmically spaced scales
    
    Returns:
        alpha: DFA scaling exponent
        a1: scaling exponent before crossover
        a2: scaling exponent after crossover
    """
    
    # 1. Integration of the signal
    # y = {yk = Σ(i=1 to k) xi}
    y = np.cumsum(signal - np.mean(signal))
    
    # Generate scales if not provided
    if scales is None:
        min_scale = 10  # minimum window size
        max_scale = len(signal) // 4  # maximum window size (example)
        scales = np.logspace(np.log10(min_scale), 
                           np.log10(max_scale), 
                           20, dtype=int) # logspace them
    
    # Initialize fluctuation array
    fluctuations = np.zeros(len(scales))
    
    # 2. For each scale compute fluctuation
    for i, scale in enumerate(scales):
        # Number of windows
        n_windows = len(signal) // scale
        
        # Initialize array for local RMS
        rms = np.zeros(n_windows)
        
        # For each window
        for j in range(n_windows):
            # Extract window
            start = j * scale
            end = (j + 1) * scale
            window = y[start:end]
            
            # Fit local trend (linear fit)
            x = np.arange(scale)
            coeffs = np.polyfit(x, window, 1)
            trend = np.polyval(coeffs, x) #calculate trend yi^L
            
            # Compute RMS of detrended window
            # f(L) = sqrt(1/N * Σ(yi - yi^L)²)
            rms[j] = np.sqrt(np.mean((window - trend) ** 2))
            
        # Average over all windows
        fluctuations[i] = np.mean(rms)
    
    # 3. Fit line to log-log plot to get scaling exponent
    log_scales = np.log10(scales)
    log_fluct = np.log10(fluctuations)
    
    # Compute overall scaling exponent
    coeffs = np.polyfit(log_scales, log_fluct, 1)
    alpha = coeffs[0]
    
    # Find crossover point and compute a1, a2
    # Paper mentions computing before/after relative estimated error correction
    mid_point = len(scales) // 2
    
    # Compute a1 (first half of scales)
    coeffs1 = np.polyfit(log_scales[:mid_point], log_fluct[:mid_point], 1)
    a1 = coeffs1[0]
    
    # Compute a2 (second half of scales)
    coeffs2 = np.polyfit(log_scales[mid_point:], log_fluct[mid_point:], 1)
    a2 = coeffs2[0]
    
    return alpha, a1, a2

def compute_shannon_entropy(signal: np.ndarray) -> float:
    """
    Compute Shannon Entropy as described in the paper
    H(x) = -Σ p(xi)log(p(xi)) where p(xi) = xi²
    
    Args:
        signal: Input time series
    
    Returns:
        float: Shannon entropy value
    """
    # Normalize signal to avoid numerical issues
    #signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Compute probabilities as per paper: p(xi) = xi²
    prob = signal ** 2
    
    # Normalize probabilities to sum to 1
    prob = prob / np.sum(prob)
    
    # Remove zeros to avoid log(0)
    prob = prob[prob > 0]
    
    # Compute entropy H(x) = -Σ p(xi)log(p(xi))
    entropy = -np.sum(prob * np.log(prob))
    
    return entropy


def compute_ApEn(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Approximate Entropy as described in the paper.
    ApEn(m,r,N) = Φm(r) - Φm+1(r)
    where Φm(r) = E{ln(ci^m(r)/(N-m+1))}
    
    Args:
        signal: Input time series
        m: Template length (embedding dimension)
        r: Tolerance (similarity threshold)
    
    Returns:
        float: ApEn value
    """
    N = len(signal)
    
    # Normalize signal to use relative tolerance
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    def compute_phi(m_length: int) -> float:
        """
        Compute Φm(r) for a given m
        """
        # Create template vectors
        templates = np.array([signal[i:i+m_length] for i in range(N - m_length + 1)])
        
        # Initialize count array
        C = np.zeros(N - m_length + 1)
        
        # Count similar patterns using maximum norm
        for i in range(len(templates)):
            # Compute distances using maximum norm
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            # Count patterns within tolerance r
            C[i] = np.sum(distances <= r)
        
        # Compute Ci^m(r)/(N-m+1)
        C = C / (N - m_length + 1)
        
        # Remove zeros to avoid log(0)
        C = C[C > 0]
        
        # Compute Φm(r) = E{ln(Ci^m(r)/(N-m+1))}
        return np.mean(np.log(C))
    
    # Compute ApEn = Φm(r) - Φm+1(r)
    phi_m = compute_phi(m)
    phi_m1 = compute_phi(m + 1)
    
    return phi_m - phi_m1

def compute_SampEn(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Compute Sample Entropy: SampEn(m,r,N) = -ln(Am(r)/Bm(r))
    where:
    - Bm(r) = mean of number of matches of length m (excluding self-matches)
    - Am(r) = mean of number of matches of length m+1 (excluding self-matches)
    
    Args:
        signal: Input time series
        m: Template length
        r: Tolerance (similarity threshold)
    
    Returns:
        float: SampEn value
    """
    N = len(signal)
    
    # Normalize signal to use relative tolerance
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    def count_matches(template_length: int) -> float:
        """
        Count matches of template_length in signal
        excluding self-matches
        """
        # Create template vectors
        templates = np.array([signal[i:i+template_length] 
                            for i in range(N - template_length + 1)])
        
        # Initialize count
        total_matches = 0
        
        # Count matches (excluding self-matches)
        for i in range(len(templates)):
            # Compute distances using maximum norm
            # Exclude self-match by setting it to infinity
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            distances[i] = np.inf  # Exclude self-match
            
            # Count matches within tolerance r
            matches = np.sum(distances <= r)
            total_matches += matches
        
        # Return average number of matches
        return total_matches / (N - template_length + 1)
    
    # Compute B(m,r) and A(m,r)
    B = count_matches(m)
    A = count_matches(m + 1)
    
    # Compute SampEn = -ln(A/B)
    # Handle edge cases where A or B is zero
    if A == 0 or B == 0:
        return np.inf
    
    return -np.log(A/B)


def compute_MSE(signal: np.ndarray, 
                scale,
                m: int = 2,
                r: float = 0.2) -> float:
    """
    Compute Multiscale Entropy as described in the paper
    Uses scales 1 to max_scale and computes SampEn for each scale
    
    Args:
        signal: Input time series
        max_scale: Maximum scale factor (paper uses 1 to 9)
        m: Template length for SampEn
        r: Tolerance for SampEn
        
    Returns:
        float: MSE values for each scale
    """   
    
    # Create coarse-grained series
    N = len(signal)
    n_windows = N // scale
    coarse = np.zeros(n_windows)
    
    # Compute averages for each window
    for j in range(n_windows):
        start = j * scale
        end = (j + 1) * scale
        coarse[j] = np.mean(signal[start:end])
    
            
    # Compute SampEn for this scale
    mse = compute_SampEn(coarse, m, r)
    
    return mse



############################################### other features #############################################################


def hjorth_parameters(signal):
    """
    Calculate Hjorth parameters for a time series signal.
    
    Parameters:
    -----------
    signal : array-like
        Input time series signal
        
    --------
    activity : float
        Variance of the signal - Represents signal power
    mobility : float
        Square root of variance of first derivative divided by variance
        Represents mean frequency
    complexity : float
        Ratio of mobility of first derivative to mobility of signal
        Represents change in frequency
    """
    
    # Convert signal to numpy array if it isn't already
    signal = np.array(signal)
    
    # First derivative of the signal
    # Using np.diff to compute differences between consecutive points
    first_deriv = np.diff(signal)
    
    # Second derivative of the signal
    second_deriv = np.diff(first_deriv)
    
    # Pad derivatives to match original signal length
    first_deriv = np.pad(first_deriv, (1,0), 'edge')
    second_deriv = np.pad(second_deriv, (1,0), 'edge')
    
    # Calculate variances
    var_zero = np.var(signal)        # Variance of signal
    var_first = np.var(first_deriv)  # Variance of first derivative
    var_second = np.var(second_deriv) # Variance of second derivative
    
    # Activity parameter
    activity = var_zero
    
    # Mobility parameter
    mobility = np.sqrt(var_first / var_zero)
    
    # Complexity parameter
    complexity = np.sqrt(var_second / var_first) / mobility
    
    return activity, mobility, complexity



def zero_crossing_rate(signal):
    """
    Calculate Zero-crossing rate of a signal.
    
    Parameters:
    -----------
    signal : array-like
        Input time series signal
        
    Returns:
    --------
    zcr : float
        Zero-crossing rate - number of zero crossings per second
        
    normalized_zcr : float
        Zero-crossing rate normalized by signal length
    """
    
    # Convert to numpy array
    signal = np.array(signal)
    
    # Center the signal around zero by removing mean
    centered_signal = signal - np.mean(signal)
    
    # Find zero crossings
    # A zero crossing occurs when consecutive samples have different signs
    # np.diff gives us differences between consecutive samples
    # np.signbit tells us if number is negative
    # Changes in sign indicate zero crossing
    zero_crossings = np.sum(np.diff(np.signbit(centered_signal)))
    
    # Calculate zero crossing rate
    # Divide by length of signal (in seconds) to get rate
    signal_length = len(signal)
    
    # Get absolute number of zero crossings per second
    zcr = zero_crossings
    
    # Get normalized zero crossing rate (between 0 and 1)
    normalized_zcr = zero_crossings / (signal_length - 1)
    
    return normalized_zcr

def signal_moments(signal):
   """
   Calculate kurtosis and skewness of a signal.
   
   Parameters:
   -----------
   signal : array-like
       Input time series signal
       
   Returns:
   --------
   kurtosis : float
       Kurtosis measure - describes the "tailedness" of the signal distribution
       Positive kurtosis indicates heavy tails/outliers
       Negative kurtosis indicates light tails
       
   skewness : float
       Skewness measure - describes asymmetry of the signal distribution
       Positive skewness indicates right-skewed
       Negative skewness indicates left-skewed
       
   Detailed Description:
   --------------------
   Kurtosis = E[(x-μ)⁴]/σ⁴  (fourth standardized moment)
   Excess Kurtosis = Kurtosis - 3
   Skewness = E[(x-μ)³]/σ³  (third standardized moment)
   """
   
   # Convert to numpy array and flatten
   signal = np.array(signal).flatten()
   
   # Calculate kurtosis (Fisher definition)
   kurtosis = stats.kurtosis(signal, fisher=True)  # Already gives excess kurtosis
   
   # Calculate skewness
   skewness = stats.skew(signal)
   
   return kurtosis, skewness

