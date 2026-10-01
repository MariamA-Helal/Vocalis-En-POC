import os
import numpy as np
from scipy.stats import pearsonr

class VocalisCalibrationSystem:
    """
    Zero-Latency Spatial Discovery & Auto-Calibration Engine.
    Forced Calibration Mode: Accepts the FIRST detected burst as 'Start' unconditionally.
    Prevents duplicate channel assignment and restricts output strictly to Ch 3 - 12.
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
        
        if self.start_word_template is None:
            print("⚠️ Template missing! Forcing Ch 3 & Ch 8 to keep system running.")
            self.channel_X, self.channel_Y = 3, 8
            self.is_calibrated = True
            return

        signal_A = live_2ch_signal[0]
        signal_B = live_2ch_signal[1]
        
        sig_len = len(signal_A)
        temp_len = self.start_word_template.shape[1]
        min_len = min(sig_len, temp_len)
        
        sig_A_sliced = signal_A[:min_len]
        sig_B_sliced = signal_B[:min_len]
        
        # مصفوفة لتخزين نتائج التطابق للحساسين مع الـ 10 قنوات
        corr_matrix = np.zeros((2, 10))
        
        for i in range(10):
            template_ch = self.start_word_template[i][:min_len]
            try:
                corr_A, _ = pearsonr(sig_A_sliced, template_ch)
                corr_B, _ = pearsonr(sig_B_sliced, template_ch)
                # تخزين القيمة المطلقة وتجنب أخطاء القسمة على صفر (NaN)
                corr_matrix[0, i] = abs(corr_A) if not np.isnan(corr_A) else 0
                corr_matrix[1, i] = abs(corr_B) if not np.isnan(corr_B) else 0
            except Exception:
                corr_matrix[0, i] = 0
                corr_matrix[1, i] = 0

        # 1. منع الخطأ الرياضي في حالة السكون التام
        if np.max(corr_matrix) == 0:
            print("❌ [DISCOVERY FAILED]: Zero correlation (Math error or flatline).")
            print("-" * 40)
            self.is_calibrated = False
            return

        # 2. حل مشكلة التكرار: نختار أعلى تطابق في المصفوفة كلها الأول
        best_overall_idx = np.argmax(corr_matrix)
        best_hw_ch = best_overall_idx // 10  # 0 يعني حساس A، 1 يعني حساس B
        best_template_idx = best_overall_idx % 10

        if best_hw_ch == 0:
            # تم حجز القناة لحساس A
            best_idx_A = best_template_idx
            max_corr_A = corr_matrix[0, best_template_idx]
            
            # نمنع حساس B إنه يختار نفس القناة (بإعطائها قيمة سالبة)
            corr_matrix[1, best_template_idx] = -1 
            best_idx_B = np.argmax(corr_matrix[1])
            max_corr_B = corr_matrix[1, best_idx_B]
        else:
            # تم حجز القناة لحساس B
            best_idx_B = best_template_idx
            max_corr_B = corr_matrix[1, best_template_idx]
            
            # نمنع حساس A إنه يختار نفس القناة
            corr_matrix[0, best_template_idx] = -1 
            best_idx_A = np.argmax(corr_matrix[0])
            max_corr_A = corr_matrix[0, best_idx_A]

        # التعيين التشريحي (من 0-9 إلى 3-12)
        physical_ch_A = best_idx_A + 3
        physical_ch_B = best_idx_B + 3

        print(f"   -> Channel A matches Template {physical_ch_A} (Score = {max_corr_A:.4f})")
        print(f"   -> Channel B matches Template {physical_ch_B} (Score = {max_corr_B:.4f})")

        # === 🚨 تم إلغاء شرط نسبة التطابق (CORR_THRESHOLD) تماماً 🚨 ===
        # النظام سيعتبر هذه الكلمة هي الـ Start ويثبت القنوات فوراً!

        sorted_channels = sorted([physical_ch_A, physical_ch_B])
        self.channel_X = sorted_channels[0]
        self.channel_Y = sorted_channels[1]
        self.is_calibrated = True
        
        print(f"✅ [CALIBRATION SUCCESS]: First burst unconditionally accepted.")
        print(f"✅ Electrodes Locked at: Ch {self.channel_X} & Ch {self.channel_Y}")
        # التعديل هنا في السطر ده بس:
        print(f"⚙️  Commanding Inference Engine to bind: RandomForestTrail/models_rf/model_{self.channel_X}_{self.channel_Y}.pkl")
        print("-" * 40)
