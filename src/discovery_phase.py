import os
import numpy as np
from scipy.stats import pearsonr

class VocalisCalibrationSystem:
    """
    Zero-Latency Spatial Discovery & Auto-Calibration Engine.
    Matches a single spoken 'Start' burst against the Universal Spatiotemporal Template
    using absolute Pearson correlation profiles (|r|) to resolve Electrode Shift.
    """
    def __init__(self):
        self.is_calibrated = False
        self.channel_X = None
        self.channel_Y = None
        self.start_word_template = None 

    def load_universal_template(self):
        """Loads pre-compiled subject-invariant grand average template."""
        template_path = 'models/universal_start.npy'
        if os.path.exists(template_path):
            self.start_word_template = np.load(template_path)
            print(f"✅ Universal 'Start' Template loaded. Shape: {self.start_word_template.shape}")
        else:
            print("❌ Error: Template not found. Run feature_extraction.py first.")

    def check_connection_watchdog(self, live_signal):
        """
        Signal Health Watchdog: Continuously monitors signal variance.
        Trips a system reset to uncalibrated factory state if inputs are open-circuit/dead.
        """
        signal_variance = np.var(live_signal)
        if signal_variance < 1e-8:
            if self.is_calibrated:
                print("\n[WATCHDOG EVENT] Hardware open-circuit or disconnected!")
                print("Resetting state: is_calibrated = False")
                self.is_calibrated = False
                self.channel_X = None
                self.channel_Y = None
            return False 
        return True 

    def discover_hardware_channels(self, live_2ch_signal):
        """
        Evaluates Pearson correlation against all 10 physiological channels.
        Applies absolute value (|r|) to maintain electrode polarity invariance.
        """
        print("\n🔍 Correlating incoming gesture against Universal Template...")
        signal_A = live_2ch_signal[0]
        signal_B = live_2ch_signal[1]
        
        # --- التعديل هنا لتوحيد طول المصفوفات ---
        # الحصول على طول الكلمة الحالية وطول القالب العالمي
        sig_len = len(signal_A)
        temp_len = self.start_word_template.shape[1]
        
        # اختيار الطول الأصغر لتجنب خطأ الـ ValueError
        min_len = min(sig_len, temp_len)
        
        # قص الإشارة الحية لتطابق الطول
        sig_A_sliced = signal_A[:min_len]
        sig_B_sliced = signal_B[:min_len]
        # -----------------------------------------
        
        best_idx_A, best_idx_B = -1, -1
        max_corr_A, max_corr_B = -1, -1
        
        for i in range(10):
            # قص القالب أيضاً ليطابق الطول
            template_ch = self.start_word_template[i][:min_len]
            
            corr_A, _ = pearsonr(sig_A_sliced, template_ch)
            corr_B, _ = pearsonr(sig_B_sliced, template_ch)
            
            # Polarity-invariant absolute correlation
            if abs(corr_A) > max_corr_A:
                max_corr_A = abs(corr_A)
                best_idx_A = i
                
            if abs(corr_B) > max_corr_B:
                max_corr_B = abs(corr_B)
                best_idx_B = i

        # Remap slice index to physical anatomical channel designations (Ch 3 - Ch 12)
        physical_ch_A = best_idx_A + 3
        physical_ch_B = best_idx_B + 3
        
        # Order-invariant lexicographical sorting
        sorted_channels = sorted([physical_ch_A, physical_ch_B])
        self.channel_X = sorted_channels[0]
        self.channel_Y = sorted_channels[1]
        self.is_calibrated = True
        
        print(f"[CALIBRATION SUCCESS] Electrodes Discovered at: Channel {self.channel_X} and Channel {self.channel_Y}")
        print(f"Commanding Inference Engine to bind: models/model_{self.channel_X}_{self.channel_Y}.pkl")
        
if __name__ == "__main__":
    vocalis = VocalisCalibrationSystem()
    vocalis.load_universal_template()
    
    # 1. Simulate Watchdog Disconnection Check
    print("\n--- Running Watchdog Check ---")
    dead_signal = np.zeros((2, 500))
    vocalis.check_connection_watchdog(dead_signal)
    
    # 2. Simulate User Uttering 'Start' with electrodes placed at Ch 5 and Ch 9
    print("\n--- Running In-Vivo Spatial Discovery Simulation ---")
    if vocalis.start_word_template is not None:
        template_len = vocalis.start_word_template.shape[1]
        simulated_live = np.array([
            vocalis.start_word_template[6] + np.random.normal(0, 0.01, template_len), # Ch 9 on Lead A
            vocalis.start_word_template[2] + np.random.normal(0, 0.01, template_len)  # Ch 5 on Lead B
        ])
        
        if vocalis.check_connection_watchdog(simulated_live):
            # Run calibration
            vocalis.discover_hardware_channels(simulated_live)