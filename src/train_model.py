import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

class VocalisModelBankTrainer:
    def __init__(self):
        # The user's brilliant spatial mapping logic
        self.spatial_mapping = {
            3: [7, 8, 9, 10, 11, 12],
            4: [7, 8, 9, 10, 11, 12],
            5: [9, 10, 11, 12],
            6: [9, 10, 11, 12],
            7: [3, 4, 11, 12],
            8: [3, 4, 11, 12],
            9: [3, 4, 5, 6],
            10: [3, 4, 5, 6],
            11: [3, 4, 5, 6, 7, 8],
            12: [3, 4, 5, 6, 7, 8]
        }
        self.unique_pairs = self._generate_unique_pairs()

    def _generate_unique_pairs(self):
        """Extracts unique non-repeating pairs based on the spatial mapping."""
        pairs = set()
        for ch1, others in self.spatial_mapping.items():
            for ch2 in others:
                # Sort the pair so (3,7) and (7,3) become the exact same tuple (3,7)
                pair = tuple(sorted([ch1, ch2]))
                pairs.add(pair)
        return sorted(list(pairs))

    def load_dataset(self):
        """Loads the full features dataset once to save memory and time."""
        print("📥 Loading the complete features dataset...")
        self.df = pd.read_csv('dataset/vocalis_features.csv')
        print(f"✅ Dataset loaded. Shape: {self.df.shape}")

    def train_pair(self, ch_X, ch_Y):
        """Trains an SVM model for a specific pair of channels."""
        # 1. Filter the dataset for these 2 channels only
        columns_to_keep = ['Label']
        for col in self.df.columns:
            if f'Ch{ch_X}_' in col or f'Ch{ch_Y}_' in col:
                columns_to_keep.append(col)
                
        filtered_df = self.df[columns_to_keep]
        X = filtered_df.drop('Label', axis=1)
        y = filtered_df['Label']
        
        # 2. Train-Test Split (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Initialize and Train the SVM Model
        model = SVC(kernel='rbf', C=10, gamma='scale') 
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluate Accuracy
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 6. Save Model and Scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/model_{ch_X}_{ch_Y}.pkl')
        joblib.dump(scaler, f'models/scaler_{ch_X}_{ch_Y}.pkl')
        
        return accuracy

    def train_all_models(self):
        """Loops through all 24 pairs and trains a model for each."""
        self.load_dataset()
        print(f"\n🚀 Starting Bank Training for {len(self.unique_pairs)} Unique Channel Pairs...\n")
        
        accuracies = []
        
        for idx, (ch_X, ch_Y) in enumerate(self.unique_pairs):
            acc = self.train_pair(ch_X, ch_Y)
            accuracies.append(acc)
            print(f"[{idx+1}/{len(self.unique_pairs)}] Model (Ch {ch_X} & Ch {ch_Y}) Trained ✅ -> Accuracy: {acc * 100:.2f}%")
            
        print(f"\n🎉 All 24 Models successfully trained and saved in the 'models' folder!")
        print(f"📊 Average System Accuracy across all placements: {sum(accuracies)/len(accuracies) * 100:.2f}%")

if __name__ == "__main__":
    bank_trainer = VocalisModelBankTrainer()
    bank_trainer.train_all_models()