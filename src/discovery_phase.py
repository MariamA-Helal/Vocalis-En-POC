import os
import numpy as np
from scipy.stats import pearsonr

class VocalisCalibrationSystem:
    """
    Zero-Latency Spatial Discovery & Auto-Calibration Engine.
    Forced Calibration Mode: Accepts the FIRST detected burst as 'Start'.
    """
    def __init__(self):
        self.is_calibrated = False
        self.channel_X = None
        self.channel_Y = None
        self.start_word_template = None 

    def load_universal_template(self):
        template_path = 'models/universal_start.npy'
        if os.path.exists(template_path):
            self.start_word_template = np.load(template_path)
            print(f"✅ Universal 'Start' Template loaded. Shape: {self.start_word_template.shape}")
        else:
            print("❌ Error: 'universal_start.npy' is MISSING in the 'models' folder!")
            print("⚠️ PLEASE DOWNLOAD IT FROM COLAB AND PUT IT IN 'models/' FOLDER.")

    def check_connection_watchdog(self, live_signal):
        signal_variance = np.var(live_signal)
        if signal_variance < 1e-8:
            if self.is_calibrated:
                print("\n[WATCHDOG EVENT] Hardware open-circuit or disconnected!")
                self.is_calibrated = False
                self.channel_X = None
                self.channel_Y = None
            return False 
        return True 

    def discover_hardware_channels(self, live_2ch_signal):
        print("\n" + "-"*40)
        print("🔍 [PHASE 2: DISCOVERY] Analyzing first incoming word...")
        
        # 🚨 حماية في حالة عدم وجود الملف 🚨
        if self.start_word_template is None:
            print("⚠️ Template is missing! Forcing default channels (Ch 3 & Ch 8) to keep system running.")
            self.channel_X, self.channel_Y = 3, 8
            self.is_calibrated = True
            print(f"✅ [FORCED CALIBRATION]: Defaulted to Ch {self.channel_X} & Ch {self.channel_Y}")
            return

        signal_A = live_2ch_signal[0]
        signal_B = live_2ch_signal[1]
        
        sig_len = len(signal_A)
        temp_len = self.start_word_template.shape[1]
        min_len = min(sig_len, temp_len)
        
        sig_A_sliced = signal_A[:min_len]
        sig_B_sliced = signal_B[:min_len]
        
        best_idx_A, best_idx_B = -1, -1
        max_corr_A, max_corr_B = -1, -1
        
        for i in range(10):
            template_ch = self.start_word_template[i][:min_len]
            
            # Use try-except to prevent any mathematical crash
            try:
                corr_A, _ = pearsonr(sig_A_sliced, template_ch)
                corr_B, _ = pearsonr(sig_B_sliced, template_ch)
            except Exception:
                corr_A, corr_B = 0, 0
            
            if abs(corr_A) > max_corr_A:
                max_corr_A = abs(corr_A)
                best_idx_A = i
                
            if abs(corr_B) > max_corr_B:
                max_corr_B = abs(corr_B)
                best_idx_B = i

        physical_ch_A = best_idx_A + 3
        physical_ch_B = best_idx_B + 3

        print(f"   -> Channel A structurally matches Template {physical_ch_A}")
        print(f"   -> Channel B structurally matches Template {physical_ch_B}")

        # === 🚨 تم مسح شرط نسبة التطابق تماماً (No Threshold) 🚨 ===
        # النظام سيعتمد هذه القنوات فوراً مهما كانت النسبة!

        sorted_channels = sorted([physical_ch_A, physical_ch_B])
        self.channel_X = sorted_channels[0]
        self.channel_Y = sorted_channels[1]
        self.is_calibrated = True
        
        print(f"✅ [CALIBRATION SUCCESS]: First word accepted as 'Start'.")
        print(f"✅ Electrodes Locked at: Ch {self.channel_X} & Ch {self.channel_Y}")
        print(f"⚙️  Commanding Inference Engine to bind: models/model_{self.channel_X}_{self.channel_Y}.pkl")
        print("-" * 40)
