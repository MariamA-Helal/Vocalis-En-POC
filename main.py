import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import os
import time
import socket
import numpy as np
import joblib

# Import core project modules
from src.discovery_phase import VocalisCalibrationSystem
from src.signal_preprocessor import LiveSignalPreprocessor

try:
    from src.tts import VocalisTTS
except ImportError:
    from src.tts_engine import VocalisTTS  

class VocalisWirelessController:
    def __init__(self, host='0.0.0.0', port=12345):
        print("=" * 60)
        print("🚀 INITIALIZING VOCALIS WIRELESS EDGE SERVER (FULL TRACE MODE)...")
        print("=" * 60)
        
        self.calibration_engine = VocalisCalibrationSystem()
        self.dsp_engine = LiveSignalPreprocessor(fs=1000.0)
        
        try:
            self.tts_engine = VocalisTTS(speech_rate=150)
        except Exception:
            print("[WARNING] TTS Engine unavailable.")
            self.tts_engine = None
            
        self.rf_model = None
        self.scaler = None
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port)) 
        
        self.calibration_engine.load_universal_template()
        print(f"📡 UDP Server is ONLINE! Listening on Port {port}...")
        
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"⚠️ TELL REHAM TO SEND DATA TO THIS IP: {local_ip}")

    def extract_realtime_features(self, word_segment):
        features = []
        for ch_idx in range(2):
            signal = word_segment[:, ch_idx]
            features.extend([
                np.sqrt(np.mean(signal**2)),                                      # RMS
                np.mean(np.abs(signal)),                                          # MAV
                np.sum(np.abs(np.diff(signal))),                                  # WL
                len(np.where(np.diff(np.signbit(signal)))[0]) / max(len(signal), 1), # ZCR
                np.sum((np.diff(signal)[:-1] * np.diff(signal)[1:] < 0) & 
                       (np.abs(np.diff(signal)[:-1] - np.diff(signal)[1:]) > 1e-5)) / max(len(signal), 1), # SSC
                np.var(signal)                                                    # VAR
            ])
        return np.array(features).reshape(1, -1)

    def load_active_rf_model(self, ch_x, ch_y):
        model_path = os.path.join('RandomForestTrail', 'models_rf', f'model_{ch_x}_{ch_y}.pkl')
        scaler_path = os.path.join('RandomForestTrail', 'models_rf', f'scaler_{ch_x}_{ch_y}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.rf_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"\n🧠 Random Forest Engine armed with: {model_path}")
            if self.tts_engine:
                self.tts_engine.speak("System is ready.")
        else:
            print(f"\n❌ Error: Model {model_path} missing. Check folder paths!")

    def run_live_server(self):
        print("\n⏳ Waiting for ESP32 Wireless Stream... (Speak now)\n")
        
        buffer_size = 1500
        live_buffer = np.zeros((buffer_size, 2))
        sample_count = 0
        
        try:
            while True:
                data, addr = self.sock.recvfrom(1024) 
                line = data.decode('utf-8').strip()
                
                if not line:
                    continue
                
                try:
                    ch1_val, ch2_val = map(float, line.split(','))
                    
                    # === التعديل السحري هنا: Scaling the Hardware Gain ===
                    ch1_val = ch1_val / 1000.0
                    ch2_val = ch2_val / 1000.0
                    # ====================================================
                    
                except ValueError:
                    continue
                
                live_buffer = np.roll(live_buffer, -1, axis=0)
                live_buffer[-1] = [ch1_val, ch2_val]
                sample_count += 1
                
                if sample_count >= 250:
                    sample_count = 0 
                    
                    if not self.calibration_engine.check_connection_watchdog(live_buffer.T):
                        sys.stdout.write("\r⚠️ Watchdog: Signal is flatlining. Check electrodes!          ")
                        sys.stdout.flush()
                        continue
                        
                    # حساب قوة الإشارة الحالية (الضوضاء أو الكلمة)
                    current_raw_rms = np.sqrt(np.mean(live_buffer**2))
                    
                    # تمرير الإشارة للـ DSP
                    word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                    
                    # 1. حالة الهدوء والضوضاء الطبيعية (طباعة في نفس السطر لتجنب الزحمة)
                    if status == "NO_BURST":
                        sys.stdout.write(f"\r📡 [LISTENING] Background Noise RMS: {current_raw_rms:.2f} | Status: {status}        ")
                        sys.stdout.flush()
                    
                    # 2. حالة التقاط إشارة (سواء اتقبلت أو اترفضت)
                    else:
                        print(f"\n\n⚙️ [PHASE 1: DSP GATE] -> Burst Detected! Raw RMS: {current_raw_rms:.2f}")
                        print(f"   -> DSP Decision: {status}")
                    
                        # 3. لو الإشارة سليمة واتقبلت
                        if status == "ACCEPTED" and word_segment is not None:
                            features = self.extract_realtime_features(word_segment)
                            print("   -> Signal is clean. Features extracted successfully:")
                            print(f"      [Ch A] RMS: {features[0][0]:.2f} | MAV: {features[0][1]:.2f} | WL: {features[0][2]:.2f}")
                            print(f"      [Ch B] RMS: {features[0][6]:.2f} | MAV: {features[0][7]:.2f} | WL: {features[0][8]:.2f}")
                            
                            # 4. مرحلة المعايرة (أول كلمة)
                            if not self.calibration_engine.is_calibrated:
                                self.calibration_engine.discover_hardware_channels(word_segment.T)
                                
                                if self.calibration_engine.is_calibrated:
                                    self.load_active_rf_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                                
                                live_buffer = np.zeros((buffer_size, 2))
                                continue

                            # 5. مرحلة الذكاء الاصطناعي
                            else:
                                print("🧠 [PHASE 3: AI INFERENCE] -> Routing features to Random Forest...")
                                scaled_features = self.scaler.transform(features)
                                prediction = self.rf_model.predict(scaled_features)[0]
                                
                                print("⭐"*40)
                                print(f" 🤖 AI PREDICTION: >>> {prediction.upper()} <<<")
                                print("⭐"*40 + "\n")
                                
                                if self.tts_engine:
                                    self.tts_engine.speak(prediction)
                                
                                live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Wireless Server safely...")
            self.sock.close()

if __name__ == "__main__":
    server = VocalisWirelessController()
    server.run_live_server()

