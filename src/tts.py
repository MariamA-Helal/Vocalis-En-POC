import pyttsx3
import threading

class VocalisTTS:
    """
    Offline Neural Text-to-Speech Engine using pyttsx3.
    Runs asynchronously to prevent blocking the real-time DAQ loop.
    Connect your Bluetooth Speaker to the laptop for amplified output.
    """
    def __init__(self, speech_rate=140):
        self.engine = pyttsx3.init()
        
        # Optimize speech rate for clear, prosthetic-style enunciation
        self.engine.setProperty('rate', speech_rate) 
        
        # Attempt to select a clear female voice (e.g., Zira on Windows)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
            
        print("🔊 Local TTS Engine initialized and bound to system audio/Bluetooth.")

    def _speak_thread(self, text):
        """Internal method to execute speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def speak(self, text):
        """
        Triggers speech in a separate background thread.
        This ensures the system immediately goes back to listening for the next word.
        """
        print(f"🗣️ [SPEAKER OUTPUT]: {text}")
        # Run in thread to avoid freezing the main loop
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

if __name__ == "__main__":
    tts = VocalisTTS()
    tts.speak("System Online. Bluetooth speaker connected.")