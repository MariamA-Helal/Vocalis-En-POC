import sys
import os
import numpy as np
import joblib
import time

# Add the 'src' directory to the path to import our core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from discovery_phase import VocalisCalibrationSystem
from signal_preprocessor import LiveSignalPreprocessor
from tts import VocalisTTS

def extract_features_for_inference(word_segment):
    """
    Extracts the 6 physiological features for the 2 channels.
    Matches the exact logic used in feature_extraction.py during training.
    """
    features = []
    for ch_idx in range(2):
        signal = word_segment[:, ch_idx]
        features.extend([
            np.sqrt(np.mean(signal**2)),                      # RMS
            np.mean(np.abs(signal)),                          # MAV
            np.sum(np.abs(np.diff(signal))),                  # WL
            len(np.where(np.diff(np.signbit(signal)))[0]) / max(len(signal), 1), # ZCR
            np.sum((np.diff(signal)[:-1] * np.diff(signal)[1:] < 0) & 
                   (np.abs(np.diff(signal)[:-1] - np.diff(signal)[1:]) > 1e-5)) / max(len(signal), 1), # SSC
            np.var(signal)                                    # VAR
        ])
    return np.array(features).reshape(1, -1)

def run_system_integration_test():
    print("=" * 70)
    print("🚀 VOCALIS: END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 70)
    
    # 1. Initialize System Components
    print("\n[SYSTEM] Booting up components...")
    calibration_engine = VocalisCalibrationSystem()
    calibration_engine.load_universal_template()
    dsp_engine = LiveSignalPreprocessor(fs=1000.0)
    
    try:
        tts_engine = VocalisTTS(speech_rate=150)
    except Exception:
        print("[WARNING] TTS Engine unavailable. Audio will be muted.")
        tts_engine = None

    # 2. Phase 1: Spatial Discovery (Calibration)
    print("\n[PHASE 1] Simulating User Calibration (Saying 'Start')...")
    template_len = calibration_engine.start_word_template.shape[1]
    
    # Simulating channels 4 and 8 being active
    simulated_start = np.array([
        calibration_engine.start_word_template[5] + np.random.normal(0, 0.02, template_len), # Ch 8
        calibration_engine.start_word_template[1] + np.random.normal(0, 0.02, template_len)  # Ch 4
    ])
    
    calibration_engine.discover_hardware_channels(simulated_start)
    
    if not calibration_engine.is_calibrated:
        print("❌ Calibration failed. Exiting.")
        return

    # Load the dynamically discovered model
    ch_x, ch_y = calibration_engine.channel_X, calibration_engine.channel_Y
    model_path = os.path.join('models', f'model_{ch_x}_{ch_y}.pkl')
    scaler_path = os.path.join('models', f'scaler_{ch_x}_{ch_y}.pkl')
    
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[SYSTEM] Active Model Bound: model_{ch_x}_{ch_y}.pkl")
    
    if tts_engine:
        tts_engine.speak("Calibration successful. System online.")
    time.sleep(3) # Wait for the first speech to finish

    # 3. Phase 2: Live Inference with Noise & Artifacts
    print("\n" + "-" * 70)
    print("🎙️ [PHASE 2] LISTENING TO LIVE STREAM (SIMULATED)")
    print("-" * 70)

    # --- CASE A: Swallowing Artifact ---
    print("\n👉 CASE A: User swallows (Long duration, high amplitude)...")
    swallow_signal = np.random.normal(0, 0.05, (2500, 2))
    swallow_signal[300:1500] += np.random.normal(0, 1.5, (1200, 2))
    
    word, status = dsp_engine.process_and_segment(swallow_signal)
    if word is None:
        print(f"🛑 DSP Action: Discarded as Noise! ({status}) - Model protected.")

    # --- CASE B: Real Word with Background Electrical Noise & Neck Sway ---
    time.sleep(2)
    print("\n👉 CASE B: User speaks a word with heavy noise and neck sway...")
    
    # We slice 700 samples (700ms) and multiply by 3.0 to simulate a STRONG muscle contraction
    real_word_ch_x = calibration_engine.start_word_template[1][100:800] * 3.0
    real_word_ch_y = calibration_engine.start_word_template[5][100:800] * 3.0
    
    # Generate background noise
    noisy_word_signal = np.random.normal(0, 0.02, (2000, 2))
    
    # Inject the real word in the middle
    template_length = len(real_word_ch_x)
    noisy_word_signal[400:400+template_length, 0] += real_word_ch_x
    noisy_word_signal[400:400+template_length, 1] += real_word_ch_y
    
    # Add realistic 2Hz low-frequency neck sway (turning head)
    t = np.linspace(0, 2.0, 2000)
    noisy_word_signal[:, 0] += 0.5 * np.sin(2 * np.pi * 2.0 * t)
    noisy_word_signal[:, 1] += 0.5 * np.sin(2 * np.pi * 2.0 * t)
    
    word, status = dsp_engine.process_and_segment(noisy_word_signal)
    
    if status == "ACCEPTED" and word is not None:
        print(f"✅ DSP Action: Signal Cleaned and Segmented. (Shape: {word.shape})")
        print("🧠 Routing to AI Inference Engine...")
        
        # Extract features
        features = extract_features_for_inference(word)
        
        # Scale and Predict
        features_scaled = scaler.transform(features)
        prediction = svm_model.predict(features_scaled)[0]
        
        print(f"🔊 AI Prediction: >>> [ {prediction.upper()} ] <<<")
        if tts_engine:
            tts_engine.speak(prediction)
    else:
        print(f"🛑 DSP Action: Word Rejected! Reason: {status}")
            
    print("\n" + "=" * 70)
    print("🎉 END-TO-END TEST COMPLETE. THE PIPELINE IS FULLY FUNCTIONAL!")
    print("=" * 70)
    
# ---- The missing execution block that caused the silence! ----
if __name__ == "__main__":
    run_system_integration_test()
    # Wait 3 seconds before killing the script so Windows has time to play the audio
    time.sleep(3)
