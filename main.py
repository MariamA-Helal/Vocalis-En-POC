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
        print("🚀 INITIALIZING VOCALIS WIRELESS EDGE SERVER (TRACE MODE)...")
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
        # 🚨 التعديل هنا: المسار الآن يشير لفولدر الـ Random Forest 🚨
        model_path = os.path.join('RandomForestTrail', 'models_rf', f'model_{ch_x}_{ch_y}.pkl')
        scaler_path = os.path.join('RandomForestTrail', 'models_rf', f'scaler_{ch_x}_{ch_y}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.rf_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"🧠 Random Forest Engine armed with: {model_path}")
            if self.tts_engine:
                self.tts_engine.speak("System is ready.")
        else:
            print(f"❌ Error: Model {model_path} missing. Check folder paths!")

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
                except ValueError:
                    continue 
                
                live_buffer = np.roll(live_buffer, -1, axis=0)
                live_buffer[-1] = [ch1_val, ch2_val]
                sample_count += 1
                
                if sample_count >= 250:
                    sample_count = 0 
                    
                    if not self.calibration_engine.check_connection_watchdog(live_buffer.T):
                        print("⚠️ Watchdog: Signal is flatlining. Check electrodes!")
                        continue
                        
                    # --- تتبع مرحلة الـ DSP ---
                    word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                    
                    if status != "NO_BURST":
                        print(f"\n⚙️ [PHASE 1: DSP GATE] -> Status: {status}")
                    
                    if status == "ACCEPTED" and word_segment is not None:
                        # 1. استخراج وطباعة السمات 
                        features = self.extract_realtime_features(word_segment)
                        print("   -> Features extracted successfully.")
                        
                        # 2. مرحلة المعايرة
                        if not self.calibration_engine.is_calibrated:
                            self.calibration_engine.discover_hardware_channels(word_segment.T)
                            
                            if self.calibration_engine.is_calibrated:
                                self.load_active_rf_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                            
                            live_buffer = np.zeros((buffer_size, 2))
                            continue

                        # 3. مرحلة الذكاء الاصطناعي والتنبؤ
                        else:
                            print("🧠 [PHASE 3: AI INFERENCE] -> Sending features to Random Forest...")
                            scaled_features = self.scaler.transform(features)
                            prediction = self.rf_model.predict(scaled_features)[0]
                            
                            print("\n" + "⭐"*20)
                            print(f" 🤖 AI PREDICTION: >>> {prediction.upper()} <<<")
                            print("⭐"*20 + "\n")
                            
                            if self.tts_engine:
                                self.tts_engine.speak(prediction)
                            
                            live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n🛑 Shutting down Wireless Server safely...")
            self.sock.close()

if __name__ == "__main__":
    server = VocalisWirelessController()
    server.run_live_server()