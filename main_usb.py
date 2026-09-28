import sys
import os
import time
import numpy as np
import joblib
import serial # Requires: pip install pyserial

# Import core project modules
from src.discovery_phase import VocalisCalibrationSystem
from src.signal_preprocessor import LiveSignalPreprocessor
from src.tts import VocalisTTS

class VocalisUSBController:
    """
    Vocalis Master Controller (Plan B: USB Serial).
    Receives high-speed sEMG data directly via USB cable.
    """
    def __init__(self, com_port='COM3', baud_rate=115200):
        print("=" * 60)
        print("🚀 INITIALIZING VOCALIS USB SERVER (PLAN B)...")
        print("=" * 60)
        
        # 1. Initialize Components
        self.calibration_engine = VocalisCalibrationSystem()
        self.dsp_engine = LiveSignalPreprocessor(fs=1000.0)
        
        try:
            self.tts_engine = VocalisTTS(speech_rate=150)
        except Exception:
            print("[WARNING] TTS Engine unavailable.")
            self.tts_engine = None
            
        self.rf_model = None
        self.scaler = None
        
        # 2. Configure USB Serial Connection
        try:
            self.ser = serial.Serial(com_port, baud_rate, timeout=1)
            # Clear any garbage data in the buffer upon connection
            self.ser.reset_input_buffer() 
            print(f"🔌 USB Connected successfully on {com_port}!")
        except Exception as e:
            print(f"❌ FATAL ERROR: Cannot open {com_port}. Check cable or close Arduino IDE Serial Monitor.")
            print(f"Details: {e}")
            sys.exit(1)
        
        self.calibration_engine.load_universal_template()

    def extract_realtime_features(self, word_segment):
        features = []
        for ch_idx in range(2):
            signal = word_segment[:, ch_idx]
            features.extend([
                np.sqrt(np.mean(signal**2)),                                      
                np.mean(np.abs(signal)),                                          
                np.sum(np.abs(np.diff(signal))),                                  
                len(np.where(np.diff(np.signbit(signal)))[0]) / max(len(signal), 1), 
                np.sum((np.diff(signal)[:-1] * np.diff(signal)[1:] < 0) & 
                       (np.abs(np.diff(signal)[:-1] - np.diff(signal)[1:]) > 1e-5)) / max(len(signal), 1), 
                np.var(signal)                                                    
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
        print("\n⏳ Waiting for ESP32 USB Stream...\n")
        
        buffer_size = 1500
        live_buffer = np.zeros((buffer_size, 2))
        sample_count = 0
        
        try:
            while True:
                # 1. Receive data packet via USB Cable
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if not line:
                        continue
                    
                    try:
                        # Parse the string into two float values
                        ch1_val, ch2_val = map(float, line.split(','))
                    except ValueError:
                        continue # Ignore bad serial lines
                    
                    # 2. Shift rolling buffer
                    live_buffer = np.roll(live_buffer, -1, axis=0)
                    live_buffer[-1] = [ch1_val, ch2_val]
                    sample_count += 1
                    
                    # 3. Process every 250 samples
                    if sample_count >= 250:
                        sample_count = 0 
                        
                        if not self.calibration_engine.check_connection_watchdog(live_buffer.T):
                            continue
                            
                        if not self.calibration_engine.is_calibrated:
                            self.calibration_engine.discover_hardware_channels(live_buffer.T)
                            if self.calibration_engine.is_calibrated:
                                self.load_active_rf_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                            continue

                        word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                        
                        if status == "ACCEPTED" and word_segment is not None:
                            features = self.extract_realtime_features(word_segment)
                            scaled_features = self.scaler.transform(features)
                            prediction = self.rf_model.predict(scaled_features)[0]
                            
                            if self.tts_engine:
                                self.tts_engine.speak(prediction)
                            
                            live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n🛑 Shutting down USB Server safely...")
            self.ser.close()

if __name__ == "__main__":
    # غداً: قومي بتغيير 'COM3' إلى الرقم الفعلي الذي يظهر لكِ في الأردوينو
    server = VocalisUSBController(com_port='COM3') 
    server.run_live_server()
