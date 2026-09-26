import sys
import os
import time
import numpy as np
import joblib
import serial # Requires: pip install pyserial

# Import Vocalis Modules
from src.discovery_phase import VocalisCalibrationSystem
from src.signal_preprocessor import LiveSignalPreprocessor
from src.tts import VocalisTTS

class VocalisMasterController:
    """
    The Main Event Loop for Vocalis.
    Reads live data from ESP32 via USB Serial, processes it, and speaks the result.
    """
    def __init__(self, serial_port='COM3', baud_rate=115200):
        print("=" * 60)
        print("🚀 INITIALIZING VOCALIS MASTER CONTROLLER...")
        print("=" * 60)
        
        self.calibration_engine = VocalisCalibrationSystem()
        self.dsp_engine = LiveSignalPreprocessor(fs=1000.0)
        
        # Audio Engine
        try:
            self.tts_engine = VocalisTTS()
        except Exception as e:
            print(f"TTS Initialization Warning: {e}")
            self.tts_engine = None

        self.svm_model = None
        self.scaler = None
        
        # Hardware Connection
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        
        # Load universal template for discovery
        self.calibration_engine.load_universal_template()

    def connect_hardware(self):
        """Attempts to open the USB Serial port to the ESP32."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"📡 Connected to ESP32 on {self.serial_port}!")
            time.sleep(2) # Wait for ESP32 to reset
            return True
        except Exception as e:
            print(f"❌ Could not connect to ESP32: {e}")
            print("Please check your COM port in Device Manager.")
            return False

    def extract_realtime_features(self, word_segment):
        """Extracts the 6 features for the segmented word."""
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

    def load_active_model(self, ch_x, ch_y):
        """Dynamically binds to the specific SVM model."""
        model_path = f"models/model_{ch_x}_{ch_y}.pkl"
        scaler_path = f"models/scaler_{ch_x}_{ch_y}.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.svm_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"🧠 AI Inference Engine armed with {model_path}")
            if self.tts_engine:
                self.tts_engine.speak("System is ready.")
        else:
            print("❌ Error: Target model missing.")

    def run_live_loop(self):
        """The Hard Real-Time Execution Loop."""
        if not self.connect_hardware():
            return

        print("\n⏳ Listening to sEMG Stream...\n")
        
        # Buffer to hold 1.5 seconds of data (1500 samples at 1000Hz)
        buffer_size = 1500
        live_buffer = np.zeros((buffer_size, 2))
        sample_count = 0
        
        try:
            while True:
                # 1. READ FROM SERIAL
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                
                try:
                    # Expecting ESP32 to print: "val1,val2"
                    ch1_val, ch2_val = map(float, line.split(','))
                except ValueError:
                    continue # Skip corrupted lines
                
                # Shift buffer and add new sample
                live_buffer = np.roll(live_buffer, -1, axis=0)
                live_buffer[-1] = [ch1_val, ch2_val]
                sample_count += 1
                
                # Only process every time we fill a decent chunk (e.g., every 500ms)
                if sample_count >= 500:
                    sample_count = 0 # Reset counter
                    
                    # 2. WATCHDOG
                    if not self.calibration_engine.check_connection_watchdog(live_buffer.T):
                        continue
                        
                    # 3. DISCOVERY PHASE
                    if not self.calibration_engine.is_calibrated:
                        print("⚠️ Uncalibrated. Please say 'Start'...")
                        self.calibration_engine.discover_hardware_channels(live_buffer.T)
                        if self.calibration_engine.is_calibrated:
                            self.load_active_model(self.calibration_engine.channel_X, self.calibration_engine.channel_Y)
                        continue

                    # 4. DSP & SEGMENTATION
                    word_segment, status = self.dsp_engine.process_and_segment(live_buffer)
                    
                    # 5. AI INFERENCE & AUDIO
                    if status == "ACCEPTED" and word_segment is not None:
                        features = self.extract_realtime_features(word_segment)
                        scaled_features = self.scaler.transform(features)
                        prediction = self.svm_model.predict(scaled_features)[0]
                        
                        if self.tts_engine:
                            self.tts_engine.speak(prediction)
                        
                        # Clear buffer to prevent double-triggering the same word
                        live_buffer = np.zeros((buffer_size, 2))

        except KeyboardInterrupt:
            print("\n🛑 System Shutting Down Safely...")
            self.ser.close()

if __name__ == "__main__":
    # NOTE: Change 'COM3' to whatever port your ESP32 is using!
    controller = VocalisMasterController(serial_port='COM3')
    controller.run_live_loop()