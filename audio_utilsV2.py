import librosa
import numpy as np
import torch
import scipy.signal

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_in_chunks(file_path, chunk_duration=10, sr=None):
    """
    Generator that yields chunks of audio data.
    
    :param file_path: Path to the audio file
    :param chunk_duration: Duration of each chunk in seconds
    :param sr: Sample rate (None for native sample rate)
    :yield: Tuple of (audio_chunk, sample_rate)
    """
    with librosa.stream(file_path, block_length=chunk_duration, frame_length=2048, hop_length=512) as stream:
        for audio_chunk in stream:
            if sr is not None and stream.sr != sr:
                audio_chunk = librosa.resample(audio_chunk, stream.sr, sr)
            yield audio_chunk, sr or stream.sr

def process_audio_in_chunks(file_path, process_func, chunk_duration=10, sr=None, device=None):
    """
    Process audio file in chunks and aggregate results.
    
    :param file_path: Path to the audio file
    :param process_func: Function to process each chunk
    :param chunk_duration: Duration of each chunk in seconds
    :param sr: Sample rate (None for native sample rate)
    :param device: torch.device to use for processing
    :return: Aggregated results
    """
    results = []
    for audio_chunk, chunk_sr in load_audio_in_chunks(file_path, chunk_duration, sr):
        chunk_tensor = torch.tensor(audio_chunk, device=device).float()
        chunk_result = process_func(chunk_tensor, chunk_sr)
        if not isinstance(chunk_result, torch.Tensor):
            raise ValueError("process_func must return a torch.Tensor")
        results.append(chunk_result)
    
    if not results:
        return None
    
    # Ensure all tensors have the same shape
    first_shape = results[0].shape
    if not all(result.shape == first_shape for result in results):
        raise ValueError("All tensors returned by process_func must have the same shape")
    
    return torch.mean(torch.stack(results), dim=0)

def set_speechbrain_local_strategy():
    """
    Set SpeechBrain to use a local strategy that doesn't rely on symlinks.
    This function should be called before any SpeechBrain operations.
    """
    from speechbrain.utils.data_utils import LocalStrategy
    
    class NoSymlinkLocalStrategy(LocalStrategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.symlink = False

    import speechbrain as sb
    sb.utils.data_utils.LOCAL_STRATEGY = NoSymlinkLocalStrategy

def separate_frequency_band(audio_signal, sample_rate, target_frequency, bandwidth=50):
    """
    Separate a specific frequency band from the audio signal.
    
    :param audio_signal: The audio signal to process
    :param sample_rate: The sample rate of the audio signal
    :param target_frequency: The center frequency of the band to separate
    :param bandwidth: The bandwidth around the target frequency
    :return: The separated frequency band signal
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    
    nyquist = 0.5 * sample_rate
    if not (0 < target_frequency < nyquist):
        raise ValueError(f"Target frequency must be between 0 and {nyquist} Hz.")
    
    low = (target_frequency - bandwidth / 2) / nyquist
    high = (target_frequency + bandwidth / 2) / nyquist
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    separated_signal = scipy.signal.lfilter(b, a, audio_signal)
    return separated_signal

