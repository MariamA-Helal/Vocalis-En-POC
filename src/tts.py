import pyttsx3
import threading
import queue

class VocalisTTS:
    """
    Offline Neural Text-to-Speech Engine using pyttsx3.
    Thread-Safe Version for Windows using a Queue-Worker architecture.
    """
    def __init__(self, speech_rate=140):
        self.speech_rate = speech_rate
        
        # Create a thread-safe queue for incoming predicted words
        self.speech_queue = queue.Queue()
        
        # Start a background worker thread that runs continuously alongside the main program
        threading.Thread(target=self._tts_worker, daemon=True).start()
        print("🔊 Local TTS Engine initialized (Queue-Worker Mode).")

    def _tts_worker(self):
        """
        This function runs continuously in the background. 
        It initializes the Windows TTS engine exactly once to prevent COM object crashes, 
        then idles waiting for words to arrive in the queue.
        """
        # 1. Initialize the engine inside the background thread (Required by Windows COM architecture)
        engine = pyttsx3.init()
        engine.setProperty('rate', self.speech_rate) 
        
        # Optimize voice selection (e.g., female voice if available)
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
            
        # 2. Infinite loop to process words as soon as they arrive in the queue
        while True:
            text = self.speech_queue.get()
            if text is None: # Signal to gracefully shut down the worker
                break
                
            try:
                # Synthesize and output the speech
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"⚠️ Audio Driver Error: {e}")
            
            # Mark the task as completed in the queue
            self.speech_queue.task_done()

    def speak(self, text):
        """
        Receives the predicted word from the AI and instantly pushes it to the queue 
        without blocking the main real-time Data Acquisition (DAQ) loop.
        """
        print(f"🗣️ [SPEAKER OUTPUT]: {text}")
        self.speech_queue.put(text)

if __name__ == "__main__":
    import time
    tts = VocalisTTS()
    tts.speak("System Online.")
    tts.speak("Testing queue.")
    time.sleep(3) # Allow time for the background thread to finish speaking in standalone test mode