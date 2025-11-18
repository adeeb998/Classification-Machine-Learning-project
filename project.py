# File: bank_ml_pipeline_download.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import wget
import zipfile
import os

# Step 1: Download dataset if not present
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
dataset_zip = "bank.zip"
dataset_csv = "bank-full.csv"

if not os.path.exists(dataset_csv):
    print("Downloading dataset...")
    wget.download(dataset_url, dataset_zip)
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("\nDataset extracted.")

# Step 2: Load dataset
data = pd.read_csv(dataset_csv, sep=';')
print("Dataset loaded. Shape:", data.shape)

# Step 3: Separate features and target
X = data.drop('y', axis=1)
y = data['y']

# Encode target
y = LabelEncoder().fit_transform(y)

# Step 4: Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Step 5: Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

print("Preprocessor ready ✅")

# Step 6: Preprocess features
X_processed = preprocessor.fit_transform(X)
print("Features preprocessed. Shape:", X_processed.shape)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Step 8: Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("RandomForest Accuracy:", acc)

