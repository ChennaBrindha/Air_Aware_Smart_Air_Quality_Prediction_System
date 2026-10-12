import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
# ADDED SelectKBest for feature selection possibility
from sklearn.feature_selection import VarianceThreshold 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
print("\n📌 Loading dataset...")
# Assuming your file path is correct
df = pd.read_csv("city_day_cleaned.csv") 
print("Dataset loaded successfully!")

print(df.head())
print("\nDataset Info:")
print(df.info())

# ---------------------------------------------------------
# 2. Drop rows where target AQI_Bucket is missing & Convert Date
# ---------------------------------------------------------
df = df.dropna(subset=["AQI_Bucket"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# NEW: Remove Date column as it's not useful without feature engineering
df = df.drop(columns=["Date"])

# Clean target column to avoid duplicates (VERY IMPORTANT)
df["AQI_Bucket"] = df["AQI_Bucket"].astype(str).str.strip().str.title()

# ---------------------------------------------------------
# 3. Split features & target
# ---------------------------------------------------------
X = df.drop(columns=["AQI_Bucket"])
y = df["AQI_Bucket"]

# REMOVE AQI because it directly determines bucket 100% accuracy
if "AQI" in X.columns:
    print("⚠ Removing AQI column ...")
    X = X.drop(columns=["AQI"])

# Encode target properly
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClasses detected:", label_encoder.classes_)
num_classes = len(label_encoder.classes_)


# ---------------------------------------------------------
# 4. Identify column types
# ---------------------------------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = []
# Ensure StationId is treated as categorical if present
if "City" in X.columns: 
    # Make sure StationId is of type object/string for OHE to work correctly
    X["City"] = X["City"].astype("object") 
    categorical_features.append("City")

# ---------------------------------------------------------
# 5. Preprocessing Pipeline
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # Optional: Feature selection using variance threshold
    # ('var_thres', VarianceThreshold(threshold=0.001)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    # Set remainder to 'passthrough' if you have other types of features (e.g., boolean)
    remainder='drop' 
)

# ---------------------------------------------------------
# 6. Train–Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 7. Optimized Base Models
# ---------------------------------------------------------
# Optimized Random Forest: Increased depth and more estimators for better coverage
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=600, 
    max_depth=30, # Increased complexity
    min_samples_split=2, 
    min_samples_leaf=1,
    n_jobs=-1, # Use all available cores
    random_state=42
)

# Optimized XGBoost: Significantly increased estimators, reduced learning rate, tuned sampling
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=1000, # Significantly increased
    learning_rate=0.05, # Reduced for better generalization (more steps)
    max_depth=7, # Slightly increased
    subsample=0.8, # Row subsampling
    colsample_bytree=0.8, # Feature subsampling
    objective="reg:squarederror", # Updated for regression
    eval_metric="rmse", # Updated for regression
    use_label_encoder=False, # Suppress warning for recent XGBoost versions
    n_jobs=-1, # Use all available cores
    random_state=42
)

# ---------------------------------------------------------
# 8. Voting Ensemble (Soft Voting)
# ---------------------------------------------------------
# Using only the two strong, optimized models
ensemble_model = VotingRegressor(
    estimators=[
        ("rf", rf_model),
        ("xgb", xgb_model)
    ],
    voting="soft",
    n_jobs=-1
)

# ---------------------------------------------------------
# 9. Full Pipeline
# ---------------------------------------------------------
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", ensemble_model)
])

# ---------------------------------------------------------
# 10. Train Model
# ---------------------------------------------------------
print("\n🚀 Training the ensemble model with optimized parameters...")
# The following line might take longer due to increased n_estimators
model_pipeline.fit(X_train, y_train) 

# ---------------------------------------------------------
# 11. Evaluate Model
# ---------------------------------------------------------
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n📈 MODEL ACCURACY:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------------------------------------------------
# 12. Save the model
# ---------------------------------------------------------
joblib.dump(model_pipeline, "aqi_model_high_accuracy.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✅ Model saved as aqi_model_high_accuracy.pkl")
print("🎉 Training complete!")