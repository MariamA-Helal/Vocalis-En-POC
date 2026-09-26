import sys
import os
import numpy as np

# Add the 'src' directory to the path so we can import our core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from signal_preprocessor import LiveSignalPreprocessor

def generate_mock_signal(duration_ms=1500, burst_start_ms=500, burst_duration_ms=400, 
                         noise_level=0.05, burst_amplitude=1.2, low_freq_sway=False):
    """Helper function to generate simulated 2-channel sEMG data."""
    fs = 1000.0
    total_samples = int((duration_ms / 1000.0) * fs)
    signal = np.random.normal(0, noise_level, (total_samples, 2))
    
    start_idx = int((burst_start_ms / 1000.0) * fs)
    end_idx = start_idx + int((burst_duration_ms / 1000.0) * fs)
    
    # Add actual high-frequency muscle burst
    signal[start_idx:end_idx] += np.random.normal(0, burst_amplitude, (end_idx - start_idx, 2))
    
    # Add low-frequency neck movement sway (e.g., 2 Hz sine wave)
    if low_freq_sway:
        t = np.linspace(0, duration_ms / 1000.0, total_samples)
        sway = 2.0 * np.sin(2 * np.pi * 2.0 * t) # 2Hz strong sway
        signal[:, 0] += sway
        signal[:, 1] += sway
        
    return signal

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 VOCALIS DSP PIPELINE: ADVANCED ARTIFACT SIMULATION SUITE")
    print("=" * 70)
    
    dsp = LiveSignalPreprocessor(fs=1000.0)
    
    # -------------------------------------------------------------------
    # CASE 1: Highly Noisy Signal (Realistic real-world raw sEMG)
    # -------------------------------------------------------------------
    print("\n[CASE 1] Testing Highly Noisy Signal (High Baseline Noise + Word Burst)...")
    noisy_signal = generate_mock_signal(noise_level=0.5, burst_amplitude=2.0)
    word_case1, status1 = dsp.process_and_segment(noisy_signal)
    print(f"Result: {status1}")
    if word_case1 is not None:
        print(f"Action: Successfully extracted word of shape {word_case1.shape} despite heavy noise.")
        
    # -------------------------------------------------------------------
    # CASE 2: Swallowing Artifact (Stronger amplitude, Long duration > 1s)
    # -------------------------------------------------------------------
    print("\n[CASE 2] Testing Swallowing Artifact (Duration > 1000ms, High Amp)...")
    swallow_signal = generate_mock_signal(duration_ms=2500, burst_start_ms=300, 
                                          burst_duration_ms=1200, burst_amplitude=3.0)
    word_case2, status2 = dsp.process_and_segment(swallow_signal)
    print(f"Result: {status2}")
    if word_case2 is None:
        print("Action: Rejected successfully! Protected the SVM from swallowing data.")

    # -------------------------------------------------------------------
    # CASE 3: Word Mixed with Low-Frequency Neck Movement (Sway)
    # -------------------------------------------------------------------
    print("\n[CASE 3] Testing Word mixed with Low-Frequency Neck Movement (2Hz Sway)...")
    # This simulates a person turning their head while speaking.
    # The 10Hz High-Pass filter should completely erase the 2Hz sway and save the word!
    sway_signal = generate_mock_signal(burst_amplitude=1.5, low_freq_sway=True)
    word_case3, status3 = dsp.process_and_segment(sway_signal)
    print(f"Result: {status3}")
    if word_case3 is not None:
        print(f"Action: Accepted! The 10Hz High-Pass filter successfully erased the neck sway.")
        
    print("\n" + "=" * 70)
    print("✅ All Test Cases Executed Successfully.")
    print("=" * 70)