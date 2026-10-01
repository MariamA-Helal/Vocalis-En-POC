import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class LiveSignalPreprocessor:
    """
    Hard Real-Time Digital Signal Processor for Dual-Channel sEMG.
    fs: Strict 1000 Hz hardware-locked sampling rate.
    """
    def __init__(self, fs=1000.0, lowcut=10.0, highcut=450.0, notch_freq=50.0, notch_q=30.0):
        self.fs = fs
        self.nyq = 0.5 * self.fs
        
        self.b_band, self.a_band = butter(4, [lowcut / self.nyq, highcut / self.nyq], btype='bandpass')
        self.b_notch, self.a_notch = iirnotch(notch_freq / self.nyq, notch_q)

    def clean_signal(self, raw_data):
        notched = filtfilt(self.b_notch, self.a_notch, raw_data, axis=0)
        cleaned = filtfilt(self.b_band, self.a_band, notched, axis=0)
        return cleaned

    def apply_tkeo(self, data):
        tkeo = np.zeros_like(data)
        tkeo[1:-1] = data[1:-1]**2 - (data[:-2] * data[2:])
        return np.abs(tkeo)

    def smooth_signal(self, data, window_size=50):
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='same')

    def calc_rms(self, signal):
        return np.sqrt(np.mean(signal**2))

    # ====== التعديلات السحرية هنا: بوابات مفتوحة على مصراعيها ======
    def process_and_segment(self, raw_buffer, 
                            threshold_multiplier=1.2,  # حساس جداً (أي حركة فوق الهدوء بـ 20% هتتقبل)
                            min_duration_ms=50.0,      # يقبل إشارات قصيرة جداً
                            max_duration_ms=3000.0,    # يقبل إشارات طويلة جداً (3 ثواني)
                            max_rms_threshold=99999.0): # مفتوح بلا حدود
        """
        Executes end-to-end DSP pipeline with physiological noise rejection gates.
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
        
        if duration_ms > max_duration_ms:
            return None, f"REJECTED_SWALLOW_DURATION ({duration_ms:.1f}ms > {max_duration_ms}ms)"
        if duration_ms < min_duration_ms:
            return None, f"REJECTED_TOO_SHORT ({duration_ms:.1f}ms < {min_duration_ms}ms)"
            
        candidate_word = cleaned[start_idx:end_idx]
        
        word_rms = self.calc_rms(candidate_word)
        if word_rms > max_rms_threshold:
            return None, f"REJECTED_HIGH_RMS_SPASM (RMS={word_rms:.2f} > {max_rms_threshold})"
            
        return candidate_word, "ACCEPTED"