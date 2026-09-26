import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

class VocalisFeatureExtractor:
    """
    Feature Extraction Pipeline for Vocalis Silent Speech Recognition.
    Processes 4 subjects across 3 sessions and 5 batches (SilentWear Corpus).
    Extracts 6 time-domain physiological descriptors per articulatory channel.
    """
    def __init__(self):
        self.subjects = ['S01', 'S02', 'S03', 'S04']
        self.sessions = [1, 2, 3] # 3 recording sessions per subject
        self.batches = [1, 2, 3, 4, 5] # 5 batches per session
        
        # Central differential laryngeal and submental channels (Ch 3 to Ch 12)
        # Sliced from 14-channel raw matrix using indices 2 through 11
        self.channels_to_keep = list(range(2, 12)) 
        
        self.all_features = []
        self.start_templates = []

    def calc_rms(self, signal):
        """Root Mean Square: Quantifies articulatory contraction energy and amplitude."""
        return np.sqrt(np.mean(signal**2))

    def calc_mav(self, signal):
        """Mean Absolute Value: Measures average muscle signal envelope."""
        return np.mean(np.abs(signal))

    def calc_wl(self, signal):
        """Waveform Length: Cumulative measure of signal complexity and waveform displacement."""
        return np.sum(np.abs(np.diff(signal)))

    def calc_zcr(self, signal):
        """Zero Crossing Rate: Evaluates dominant frequency shifts across zero-potential baseline."""
        zero_crosses = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crosses) / len(signal)

    def calc_ssc(self, signal, threshold=1e-5):
        """Slope Sign Changes: Captures neuromuscular micro-twitches and firing rate shifts."""
        diff = np.diff(signal)
        ssc = np.sum((diff[:-1] * diff[1:] < 0) & (np.abs(diff[:-1] - diff[1:]) > threshold))
        return ssc / len(signal)

    def calc_var(self, signal):
        """Variance: Measures the power dispersion of the active contraction burst."""
        return np.var(signal)

    def build_dataset(self):
        """Iterates through all subjects, sessions, and batches to extract data."""
        print("🚀 Starting Data Processing & Feature Extraction Pipeline...")
        
        for subject in self.subjects:
            for session in self.sessions:
                for batch in self.batches:
                    print(f"\n📥 Fetching data for {subject} - Session {session} - Batch {batch}...")
                    try:
                        # Download specific silent speech HDF5 file
                        file_path = hf_hub_download(
                            repo_id="PulpBio/SilentWear", 
                            filename=f"data_raw_and_filt/{subject}/silent/sess_{session}_batch_{batch}.h5", 
                            repo_type="dataset"
                        )
                        
                        # Load HDF5 file into Pandas
                        df = pd.read_hdf(file_path, key="emg")
                        
                        # Segment contiguous identical articulatory word blocks
                        df['block'] = (df['Label_str'] != df['Label_str'].shift()).cumsum()
                        grouped = df.groupby('block')
                        
                        for _, group in grouped:
                            word_label = group['Label_str'].iloc[0]
                            # Retain only the 10 central differential channels
                            signal_matrix = group.iloc[:, 2:12].values 
                            
                            # Cache isolated "Start" words for the Universal Calibration Template
                            if word_label.lower() == 'start':
                                self.start_templates.append(signal_matrix.T) 
                            
                            word_features = {'Label': word_label, 'Subject': subject}
                            
                            for ch_idx in range(10):
                                ch_signal = signal_matrix[:, ch_idx]
                                physical_ch_number = ch_idx + 3
                                
                                # Compute the 6 mathematical descriptors
                                word_features[f'Ch{physical_ch_number}_RMS'] = self.calc_rms(ch_signal)
                                word_features[f'Ch{physical_ch_number}_MAV'] = self.calc_mav(ch_signal)
                                word_features[f'Ch{physical_ch_number}_WL']  = self.calc_wl(ch_signal)
                                word_features[f'Ch{physical_ch_number}_ZCR'] = self.calc_zcr(ch_signal)
                                word_features[f'Ch{physical_ch_number}_SSC'] = self.calc_ssc(ch_signal)
                                word_features[f'Ch{physical_ch_number}_VAR'] = self.calc_var(ch_signal)
                                
                            self.all_features.append(word_features)
                            
                    except Exception as e:
                        print(f"⚠️ Error processing {subject} (Sess {session}, Batch {batch}): {e}")

        self.save_outputs()

    def save_outputs(self):
        """Saves tabular features to CSV and the Universal Calibration Template to NPY."""
        print("\n💾 Serializing extracted features...")
        features_df = pd.DataFrame(self.all_features)
        
        # Save CSV features
        os.makedirs('dataset', exist_ok=True)
        csv_path = 'dataset/vocalis_features.csv'
        features_df.to_csv(csv_path, index=False)
        print(f"✅ Features saved successfully to: {csv_path} (Shape: {features_df.shape})")
        
        # Compile and serialize the subject-invariant Universal 'Start' Template
        if self.start_templates:
            min_len = min([temp.shape[1] for temp in self.start_templates])
            truncated_templates = [temp[:, :min_len] for temp in self.start_templates]
            
            # Compute grand average waveform across all subjects
            universal_template = np.mean(truncated_templates, axis=0)
            
            os.makedirs('models', exist_ok=True)
            template_path = 'models/universal_start.npy'
            np.save(template_path, universal_template)
            print(f"✅ Universal 'Start' template saved to: {template_path} (Shape: {universal_template.shape})")

if __name__ == "__main__":
    extractor = VocalisFeatureExtractor()
    extractor.build_dataset()