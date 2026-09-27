import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

class XGBoostBankTrainer:
    def __init__(self):
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
        pairs = set()
        for ch1, others in self.spatial_mapping.items():
            for ch2 in others:
                pair = tuple(sorted([ch1, ch2]))
                pairs.add(pair)
        return sorted(list(pairs))

    def load_dataset(self):
        print("📥 Loading features dataset for XGBoost...")
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'vocalis_features.csv')
        self.df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded. Shape: {self.df.shape}")

    def train_pair(self, ch_X, ch_Y):
        columns_to_keep = ['Label']
        for col in self.df.columns:
            if f'Ch{ch_X}_' in col or f'Ch{ch_Y}_' in col:
                columns_to_keep.append(col)
                
        filtered_df = self.df[columns_to_keep]
        X = filtered_df.drop('Label', axis=1)
        y = filtered_df['Label']
        
        # XGBoost requires target labels to be numeric (0, 1, 2...), not strings ('Up', 'Down')
        # So we encode them first.
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize XGBoost Classifier (Using mostly default hyper-parameters optimized for tabular data)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        save_dir = os.path.join(os.path.dirname(__file__), 'models_xgb')
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, os.path.join(save_dir, f'model_{ch_X}_{ch_Y}.pkl'))
        joblib.dump(scaler, os.path.join(save_dir, f'scaler_{ch_X}_{ch_Y}.pkl'))
        # Save LabelEncoder because we will need it to decode numbers back to words later
        joblib.dump(le, os.path.join(save_dir, f'encoder_{ch_X}_{ch_Y}.pkl'))
        
        return accuracy

    def train_all_models(self):
        self.load_dataset()
        print(f"\n🚀 Starting XGBoost Training for {len(self.unique_pairs)} Pairs...\n")
        
        accuracies = []
        for idx, (ch_X, ch_Y) in enumerate(self.unique_pairs):
            acc = self.train_pair(ch_X, ch_Y)
            accuracies.append(acc)
            print(f"🔥 XGB Model (Ch {ch_X} & Ch {ch_Y}) -> Accuracy: {acc * 100:.2f}%")
            
        print(f"\n🎉 All XGB Models saved in 'XGBoostTrial/models_xgb' folder!")
        print(f"📊 XGB Average System Accuracy: {sum(accuracies)/len(accuracies) * 100:.2f}%")

if __name__ == "__main__":
    xgb_trainer = XGBoostBankTrainer()
    xgb_trainer.train_all_models()