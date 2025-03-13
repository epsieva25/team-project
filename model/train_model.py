import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib


file_path = "network_analysis_dataset.xlsx"

try:
    df = pd.read_excel(file_path, engine="openpyxl")
    print("\n✅ Dataset loaded successfully!")
except FileNotFoundError:
    print(f"\n❌ Error: File '{file_path}' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    exit()

# Drop unnecessary columns
drop_columns = ['public_ip_info', 'isp', 'vpn']
df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

# Encode target variable
if 'connection_status' in df.columns:
    label_encoder = LabelEncoder()
    df['connection_status'] = label_encoder.fit_transform(df['connection_status'])
else:
    print("\n❌ Error: 'connection_status' column not found in dataset.")
    exit()

# Feature Engineering
df['download_upload_ratio'] = df['download_speed_mbps'] / (df['upload_speed_mbps'] + 1)
df['latency_per_mb'] = df['ping_ms'] / (df['sent_mb'] + df['received_mb'] + 1)
df['speed_variation'] = df['download_speed_mbps'] - df['upload_speed_mbps']
df['ping_to_latency_ratio'] = df['ping_ms'] / (df['real_time_latency_ms'] + 1)
df['total_data_transfer'] = df['sent_mb'] + df['received_mb']

# Define features (X) and target variable (y)
X = df.drop(columns=['connection_status'])
y = df['connection_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to XGBoost DMatrix format for early stopping
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 50,  # Reduced for faster training
    'learning_rate': 0.03,  # Lowered for better generalization
    'max_depth': 3,  # Prevents overfitting
    'min_child_weight': 3,  # Avoids learning noise
    'reg_alpha': 1.0,  # L1 Regularization
    'reg_lambda': 1.0,  # L2 Regularization
    'subsample': 0.8,  # Uses 80% of training samples per tree
    'colsample_bytree': 0.8,  # Uses 80% of features per tree
    'eval_metric': 'logloss'  # Standard evaluation metric for binary classification
}

# Train Model with Early Stopping
evals = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,  # Maximum boosting rounds
    evals=evals,
    early_stopping_rounds=10,  # Stops if validation loss doesn't improve
    verbose_eval=True
)

# Predictions
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", round(accuracy * 100, 2), "%")
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Analysis
feature_importance = xgb_model.get_score(importance_type='weight')
important_features = pd.Series(feature_importance).sort_values(ascending=False)

print("\n📌 Top Features:\n", important_features.head(10))

# Save the trained model and scaler
joblib.dump(xgb_model, 'optimized_network_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Optimized Model and scaler saved successfully!")
