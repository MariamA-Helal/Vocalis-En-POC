import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class LiveSignalPreprocessor:
    """
    Hard Real-Time Digital Signal Processor for Dual-Channel sEMG.
    fs: Strict 1000 Hz hardware-locked sampling rate.
    Implements cascaded digital filtering, TKEO burst detection, and physiological heuristic noise gates.
    """
    def __init__(self, fs=1000.0, lowcut=10.0, highcut=450.0, notch_freq=50.0, notch_q=30.0):
        self.fs = fs
        self.nyq = 0.5 * self.fs
        
        # Pre-compute filter coefficients once in memory for deterministic low-latency execution
        self.b_band, self.a_band = butter(4, [lowcut / self.nyq, highcut / self.nyq], btype='bandpass')
        self.b_notch, self.a_notch = iirnotch(notch_freq / self.nyq, notch_q)

    def clean_signal(self, raw_data):
        """Applies cascaded 50 Hz Notch filter followed by 10-450 Hz 4th-Order Butterworth Bandpass."""
        notched = filtfilt(self.b_notch, self.a_notch, raw_data, axis=0)
        cleaned = filtfilt(self.b_band, self.a_band, notched, axis=0)
        return cleaned

    def apply_tkeo(self, data):
        """Teager-Kaiser Energy Operator (TKEO): y[n] = x[n]^2 - x[n-1] * x[n+1]"""
        tkeo = np.zeros_like(data)
        tkeo[1:-1] = data[1:-1]**2 - (data[:-2] * data[2:])
        return np.abs(tkeo)

    def smooth_signal(self, data, window_size=50):
        """Calculates instantaneous energy envelope via moving average."""
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='same')

    def calc_rms(self, signal):
        """Calculates Root Mean Square power of candidate segmented word."""
        return np.sqrt(np.mean(signal**2))

    def process_and_segment(self, raw_buffer, 
                            threshold_multiplier=2.5,  # كان 4.0 (قللناه عشان يلقط الكلمة أسهل)
                            min_duration_ms=100.0,     # كان 180 (عشان يقبل الكلمات السريعة)
                            max_duration_ms=2000.0,    # كان 850 (وسعناه جداً لثانيتين)
                            max_rms_threshold=50.0):   # كان 4.5 (رفعناه جداً عشان ميقصش الإشارة القوية)
        """
        Executes end-to-end DSP pipeline with physiological noise rejection gates.
        
        Returns:
            segmented_word: Clean 2-channel word array (if accepted), or None.
            status: Diagnostic status string.
        """
        cleaned = self.clean_signal(raw_buffer)
        tkeo_energy = self.apply_tkeo(cleaned)
        smoothed = self.smooth_signal(tkeo_energy)
        
        baseline_noise = np.mean(smoothed[:150])
        threshold = baseline_noise * threshold_multiplier
        
        active_indices = np.where(smoothed > threshold)[0]
        if len(active_indices) == 0:
            return None, "NO_BURST"
            
        start_idx = active_indices[0]
        end_idx = active_indices[-1]
        
        margin_samples = int(0.070 * self.fs)
        start_idx = max(0, start_idx - margin_samples)
        end_idx = min(len(cleaned), end_idx + margin_samples)
        
        duration_ms = ((end_idx - start_idx) / self.fs) * 1000.0
        
        # HEURISTIC GATE 1 - Duration Check
        if duration_ms > max_duration_ms:
            return None, f"REJECTED_SWALLOW_DURATION ({duration_ms:.1f}ms > {max_duration_ms}ms)"
        if duration_ms < min_duration_ms:
            return None, f"REJECTED_TOO_SHORT ({duration_ms:.1f}ms < {min_duration_ms}ms)"
            
        candidate_word = cleaned[start_idx:end_idx]
        
        # HEURISTIC GATE 2 - Amplitude Power Check
        word_rms = self.calc_rms(candidate_word)
        if word_rms > max_rms_threshold:
            return None, f"REJECTED_HIGH_RMS_SPASM (RMS={word_rms:.2f} > {max_rms_threshold})"
            
        return candidate_word, "ACCEPTED"