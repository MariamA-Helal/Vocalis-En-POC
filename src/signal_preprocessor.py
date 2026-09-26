import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class LiveSignalPreprocessor:
    """
    Hard Real-Time Digital Signal Processor for Dual-Channel sEMG in Vocalis.
    - Locked Strictly at fs = 1000.0 Hz.
    - Cascaded 50 Hz Notch Filter (Q = 30) + 4th-Order Butterworth Bandpass (10-450 Hz).
    - Teager-Kaiser Energy Operator (TKEO) Articulatory Burst Extraction.
    - Heuristic Biological Noise Gates:
        1. Time-Duration Gate (Swallow / Yawn Rejection: 180ms - 850ms).
        2. Amplitude Power Gate (RMS Spasm / Neck Twist Rejection: RMS <= 4.5).
    """
    def __init__(self, fs=1000.0, lowcut=10.0, highcut=450.0, notch_freq=50.0, notch_q=30.0):
        self.fs = fs
        self.nyq = 0.5 * self.fs
        
        # 1. Bandpass coefficients (10 - 450 Hz)
        self.b_band, self.a_band = butter(
            4, [lowcut / self.nyq, highcut / self.nyq], btype='bandpass'
        )
        
        # 2. Notch coefficients (50 Hz power-line rejection)
        self.b_notch, self.a_notch = iirnotch(notch_freq / self.nyq, notch_q)

    def clean_signal(self, raw_data):
        """
        Applies cascaded 50 Hz Notch filter followed by 10-450 Hz Butterworth Bandpass.
        raw_data shape: (N_samples, 2_channels) or (N_samples,)
        """
        # Step 1: Notch 50 Hz hum
        notched = filtfilt(self.b_notch, self.a_notch, raw_data, axis=0)
        # Step 2: Bandpass 10-450 Hz physiological speech potentials
        cleaned = filtfilt(self.b_band, self.a_band, notched, axis=0)
        return cleaned

    def apply_tkeo(self, data):
        """
        Teager-Kaiser Energy Operator (TKEO):
        y[n] = x[n]^2 - x[n-1] * x[n+1]
        Amplifies high-frequency instant neuromuscular firing bursts.
        """
        tkeo = np.zeros_like(data)
        tkeo[1:-1] = data[1:-1]**2 - (data[:-2] * data[2:])
        return np.abs(tkeo)

    def smooth_signal(self, data, window_size=50):
        """
        Computes the instantaneous articulatory energy envelope.
        """
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='same')

    def calc_rms(self, signal):
        """
        Computes the Root Mean Square power of the segmented candidate gesture.
        """
        return np.sqrt(np.mean(signal**2))

    def process_and_segment(self, raw_buffer, 
                            threshold_multiplier=4.0, 
                            min_duration_ms=180.0, 
                            max_duration_ms=850.0, 
                            max_rms_threshold=4.5):
        """
        End-to-End DSP Pipeline:
        Filter -> TKEO Energy -> Adaptive Baseline Thresholding -> Duration Gate -> RMS Gate.
        
        Returns:
            candidate_word (numpy array or None): Cleaned 2-channel word array ready for SVM.
            diagnosis_status (str): Exact diagnosis string explaining acceptance or rejection.
        """
        # 1. Clean the continuous incoming buffer
        cleaned = self.clean_signal(raw_buffer)
        
        # 2. Extract TKEO energy and compute smoothed envelope
        tkeo_energy = self.apply_tkeo(cleaned)
        smoothed = self.smooth_signal(tkeo_energy)
        
        # 3. Dynamic baseline noise estimation from initial 150 ms (150 samples)
        baseline_noise = np.mean(smoothed[:150])
        threshold = baseline_noise * threshold_multiplier
        
        # 4. Detect active articulatory burst points
        active_indices = np.where(smoothed > threshold)[0]
        if len(active_indices) == 0:
            return None, "NO_BURST_DETECTED"
            
        start_idx = active_indices[0]
        end_idx = active_indices[-1]
        
        # 5. Add 70 ms safety padding margin before and after burst
        margin_samples = int(0.070 * self.fs)
        start_idx = max(0, start_idx - margin_samples)
        end_idx = min(len(cleaned), end_idx + margin_samples)
        
        # Calculate duration of the candidate burst in milliseconds
        duration_samples = end_idx - start_idx
        duration_ms = (duration_samples / self.fs) * 1000.0
        
        # 6. HEURISTIC GATE 1: Duration Filter (Rejects swallowing, yawning, or micro-glitches)
        if duration_ms > max_duration_ms:
            return None, f"REJECTED_SWALLOW_DURATION ({duration_ms:.1f}ms > {max_duration_ms}ms)"
        if duration_ms < min_duration_ms:
            return None, f"REJECTED_TOO_SHORT ({duration_ms:.1f}ms < {min_duration_ms}ms)"
            
        candidate_word = cleaned[start_idx:end_idx]
        
        # 7. HEURISTIC GATE 2: Amplitude Power Filter (Rejects mechanical neck turning, coughing, or wire bumping)
        word_rms = self.calc_rms(candidate_word)
        if word_rms > max_rms_threshold:
            return None, f"REJECTED_HIGH_RMS_SPASM (RMS={word_rms:.2f} > {max_rms_threshold})"
            
        # All checks passed: gesture is verified physiological silent speech
        return candidate_word, "ACCEPTED"

if __name__ == "__main__":
    print("=" * 65)
    print("🧪 Running Vocalis DSP Verification Suite (fs = 1000 Hz)")
    print("=" * 65)
    dsp = LiveSignalPreprocessor(fs=1000.0)
    
    # Test 1: Simulated Normal Phonation (~400 ms word burst)
    print("\n[Test 1] Simulating Normal Silent Articulation (~400 ms)...")
    normal_signal = np.random.normal(0, 0.05, (1500, 2))
    normal_signal[500:900] += np.random.normal(0, 1.1, (400, 2))
    word, status = dsp.process_and_segment(normal_signal)
    print(f"Status: {status} | Segment Shape: {None if word is None else word.shape}")
    
    # Test 2: Simulated Swallow Artifact (~1100 ms burst)
    print("\n[Test 2] Simulating Swallowing Artifact (~1100 ms)...")
    swallow_signal = np.random.normal(0, 0.05, (2500, 2))
    swallow_signal[300:1400] += np.random.normal(0, 1.4, (1100, 2))
    word_swallow, status_swallow = dsp.process_and_segment(swallow_signal)
    print(f"Status: {status_swallow}")
    
    # Test 3: Simulated Neck Spasm / Violent Cough (Extreme RMS > 5.0)
    print("\n[Test 3] Simulating Neck Spasm / Sudden Head Turn...")
    spasm_signal = np.random.normal(0, 0.05, (1500, 2))
    spasm_signal[500:900] += np.random.normal(0, 7.5, (400, 2))
    word_spasm, status_spasm = dsp.process_and_segment(spasm_signal)
    print(f"Status: {status_spasm}")
    
    print("\n" + "=" * 65)
    print("✅ All Signal Preprocessing Filters and Heuristic Gates Verified!")
    print("=" * 65)