import sys
import os
import time
import socket
import numpy as np
import joblib

# Import core project modules
from src.discovery_phase import VocalisCalibrationSystem
from src.signal_preprocessor import LiveSignalPreprocessor
from src.tts import VocalisTTS

class VocalisWirelessController:
    """
    Vocalis Wireless Master Controller (Local Edge AI Server).
    Acts as a high-speed UDP server receiving wireless sEMG data from the ESP32,
    processes it using the Random Forest engine, and articulates it via Bluetooth.
    """
    def __init__(self, host='0.0.0.0', port=12345):
        print("=" * 60)
        print("🚀 INITIALIZING VOCALIS WIRELESS EDGE SERVER...")
        print("=" * 60)
        
        # 1. Initialize System Components
        self.calibration_engine = VocalisCalibrationSystem()
        self.dsp_engine = LiveSignalPreprocessor(fs=1000.0)
        
        try:
            self.tts_engine = VocalisTTS(speech_rate=150)
        except Exception:
            print("[WARNING] TTS Engine unavailable.")
            self.tts_engine = None
            
        self.rf_model = None
        self.scaler = None
        
        # 2. Configure Wireless Communication (UDP Socket)
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to allow receiving data from any device on the network
        self.sock.bind((self.host, self.port)) 
        
        # Load the universal calibration template
        self.calibration_engine.load_universal_template()
        print(f"📡 UDP Server is ONLINE! Listening on Port {self.port}...")
        
        # Fetch and display the Laptop's Local IP address for the ESP32 programmer
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"⚠️ TELL REHAM TO SEND DATA TO THIS IP: {local_ip}")

    def extract_realtime_features(self, word_segment):
        """Extracts the 6 time-domain physiological features from the segmented word."""
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
        """Dynamically loads the Random Forest model matching the discovered physical channels."""
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
        """The Main Infinite Loop for high-speed wireless reception and processing."""
        print("\n⏳ Waiting for ESP32 Wireless Stream...\n")
        
        # Initialize a rolling buffer to hold 1.5 seconds of data (1500 samples)
        buffer_size = 1500
        live_buffer = np.zeros((buffer_size, 2))
        sample_count = 0
        
        try:
            while True:
                # 1. Receive data packet via Wi-Fi (UDP)
                data, addr = self.sock.recvfrom(1024) 
                line = data.decode('utf-8').strip()
                
                if not line:
                    continue
                
                try:
                    # Parse the string into two float values
                    ch1_val, ch2_val = map(float, line.split(','))
                except ValueError:
                    continue 
                
                # 2. Shift the rolling buffer and append the new reading
                live_buffer = np.roll(live_buffer, -1, axis=0)
                live_buffer[-1] = [ch1_val, ch2_val]
                sample_count += 1
                
                # 3. Process the data every 250 milliseconds
                if sample_count >= 250:
                    sample_count = 0 
                    
                    # Connection Watchdog (Check if electrodes fell off)
                    if not self.calibration_engine.check_connection_watchdog(live_buffer.T):
                        continue
                        
                    # 💡 التعديل هنا: نمرر الإشارة للـ DSP أولاً لاكتشاف وتجاهل الضوضاء
                    word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                    
                    # إذا التقط الـ DSP كلمة حقيقية نظيفة
                    if status == "ACCEPTED" and word_segment is not None:
                        
                        # الحالة الأولى: النظام لم تتم معايرته بعد
                        if not self.calibration_engine.is_calibrated:
                            print("\n🎯 Valid word detected! Checking if it matches 'Start'...")
                            self.calibration_engine.discover_hardware_channels(word_segment.T)
                            
                            if self.calibration_engine.is_calibrated:
                                self.load_active_rf_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                            else:
                                print("🔄 Waiting for a clearer 'Start' command...")
                            
                            # تفريغ الـ Buffer في كلتا الحالتين
                            live_buffer = np.zeros((buffer_size, 2))
                            continue
                            
                        # الحالة الثانية: النظام تمت معايرته (تشغيل الموديل)
                        else:
                            features = self.extract_realtime_features(word_segment)
                            scaled_features = self.scaler.transform(features)
                            prediction = self.rf_model.predict(scaled_features)[0]
                            
                            # --- 🚨 اللوجيك الجديد: طباعة الكلمة بوضوح في الترمينال 🚨 ---
                            print("\n" + "="*40)
                            print(f" 🤖 AI PREDICTION: >>> {prediction.upper()} <<<")
                            print("="*40 + "\n")
                            # -------------------------------------------------------------
                            
                            if self.tts_engine:
                                self.tts_engine.speak(prediction)
                            
                            live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n🛑 Shutting down Wireless Server safely...")
            self.sock.close()

if __name__ == "__main__":
    server = VocalisWirelessController()
    server.run_live_server()