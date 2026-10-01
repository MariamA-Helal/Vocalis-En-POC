import sys
import os
import time
import numpy as np
import joblib
import serial # Requires: pip install pyserial

from src.discovery_phase import VocalisCalibrationSystem
from src.signal_preprocessor import LiveSignalPreprocessor
from src.tts_engine import VocalisTTS

class VocalisUSBController:
    def __init__(self, com_port='COM3', baud_rate=115200):
        print("=" * 60)
        print("🚀 INITIALIZING VOCALIS USB SERVER (TRANSPARENT DEBUG MODE)...")
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
        
        try:
            self.ser = serial.Serial(com_port, baud_rate, timeout=1)
            self.ser.reset_input_buffer() 
            print(f"🔌 USB Connected successfully on {com_port}!")
        except Exception as e:
            print(f"❌ FATAL ERROR: Cannot open {com_port}. Check your Cable or Port Number!")
            sys.exit(1)
        
        self.calibration_engine.load_universal_template()

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
        model_path = os.path.join('models', f'model_{ch_x}_{ch_y}.pkl')
        scaler_path = os.path.join('models', f'scaler_{ch_x}_{ch_y}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.rf_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"🧠 Random Forest Engine armed with: model_{ch_x}_{ch_y}.pkl")
            if self.tts_engine:
                self.tts_engine.speak("System is ready.")
        else:
            print(f"❌ Error: Model {model_path} missing.")

    def run_live_server(self):
        print("\n⏳ Listening to ESP32 USB Stream... (Speak now)\n")
        buffer_size = 1500
        live_buffer = np.zeros((buffer_size, 2))
        sample_count = 0
        
        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    
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
                            
                        word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                        
                        # --- 🚨 طباعة حالة الإشارة بدقة 🚨 ---
                        if status != "NO_BURST":
                            print(f"\n🔎 [DSP GATE]: {status}")
                        
                        if status == "ACCEPTED" and word_segment is not None:
                            # 1. استخراج وطباعة السمات لتراجعيها بعينك
                            features = self.extract_realtime_features(word_segment)
                            print("-" * 50)
                            print("📊 EXTRACTED FEATURES FOR THIS WORD:")
                            print(f"Ch_A -> RMS: {features[0][0]:.2f} | MAV: {features[0][1]:.2f} | WL: {features[0][2]:.2f}")
                            print(f"Ch_B -> RMS: {features[0][6]:.2f} | MAV: {features[0][7]:.2f} | WL: {features[0][8]:.2f}")
                            print("-" * 50)

                            # 2. مرحلة المعايرة
                            if not self.calibration_engine.is_calibrated:
                                print("🎯 Attempting 'Start' Calibration...")
                                self.calibration_engine.discover_hardware_channels(word_segment.T)
                                
                                if self.calibration_engine.is_calibrated:
                                    self.load_active_rf_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                                
                                live_buffer = np.zeros((buffer_size, 2))
                                continue

                            # 3. مرحلة الذكاء الاصطناعي والتنبؤ
                            else:
                                scaled_features = self.scaler.transform(features)
                                prediction = self.rf_model.predict(scaled_features)[0]
                                
                                print("\n" + "="*40)
                                print(f" 🤖 AI PREDICTION: >>> {prediction.upper()} <<<")
                                print("="*40 + "\n")
                                
                                if self.tts_engine:
                                    self.tts_engine.speak(prediction)
                                
                                live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n🛑 Shutting down USB Server safely...")
            self.ser.close()

if __name__ == "__main__":
    # ⚠️ تأكدي من كتابة رقم الـ COM الصحيح هنا ⚠️
    server = VocalisUSBController(com_port='COM6') 
    server.run_live_server()
