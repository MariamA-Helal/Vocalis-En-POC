import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class RandomForestBankTrainer:
    def __init__(self):
        self.spatial_mapping = {
            3: [7, 8, 9, 10, 11, 12], 4: [7, 8, 9, 10, 11, 12],
            5: [9, 10, 11, 12], 6: [9, 10, 11, 12],
            7: [3, 4, 11, 12], 8: [3, 4, 11, 12],
            9: [3, 4, 5, 6], 10: [3, 4, 5, 6],
            11: [3, 4, 5, 6, 7, 8], 12: [3, 4, 5, 6, 7, 8]
        }
        self.unique_pairs = self._generate_unique_pairs()

    def _generate_unique_pairs(self):
        pairs = set()
        for ch1, others in self.spatial_mapping.items():
            for ch2 in others:
                pair = tuple(sorted([ch1, ch2]))
                pairs.add(pair)
        return sorted(list(pairs))

    def load_dataset(self):
        print("📥 Loading features dataset...")
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'vocalis_features.csv')
        self.df = pd.read_csv(dataset_path)
        
        initial_shape = self.df.shape
        
        # === 🚨 التعديل الأول: 4 كلمات فقط للعرض العملي (POC) 🚨 ===
        # ده هيرفع الـ Accuracy جداً وهيخلي الموديل يقدر يفرق بينهم بقناتين بس
        valid_commands = ['up', 'down', 'start', 'stop']
        self.df = self.df[self.df['Label'].astype(str).str.lower().isin(valid_commands)]
        
        print(f"✅ Dataset loaded. Shape dropped from {initial_shape} to {self.df.shape} (Focused 4-Word POC)")

    def train_pair(self, ch_X, ch_Y):
        columns_to_keep = ['Label']
        for col in self.df.columns:
            if f'Ch{ch_X}_' in col or f'Ch{ch_Y}_' in col:
                columns_to_keep.append(col)
                
        filtered_df = self.df[columns_to_keep]
        X = filtered_df.drop('Label', axis=1)
        y = filtered_df['Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # === 🚨 التعديل الثاني: منع الـ Overfitting وحل مشكلة الـ 75 ميجا 🚨 ===
        # max_depth=10 هيخلي حجم الموديل صغير جداً (GitHub مش هيعترض)
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1) 
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        save_dir = os.path.join(os.path.dirname(__file__), 'models_rf')
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, os.path.join(save_dir, f'model_{ch_X}_{ch_Y}.pkl'))
        joblib.dump(scaler, os.path.join(save_dir, f'scaler_{ch_X}_{ch_Y}.pkl'))
        
        return accuracy

    def train_all_models(self):
        self.load_dataset()
        for idx, (ch_X, ch_Y) in enumerate(self.unique_pairs):
            acc = self.train_pair(ch_X, ch_Y)
            print(f"🌲 RF Model & Scaler (Ch {ch_X} & Ch {ch_Y}) Generated -> Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    rf_trainer = RandomForestBankTrainer()
    rf_trainer.train_all_models()
