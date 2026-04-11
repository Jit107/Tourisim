# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

# Updated dataset paths to point to the tourism project's processed data
Xtrain_path = "hf://datasets/jit0107/Tourism/Xtrain.csv"
Xtest_path = "hf://datasets/jit0107/Tourism/Xtest.csv"
ytrain_path = "hf://datasets/jit0107/Tourism/ytrain.csv"
ytest_path = "hf://datasets/jit0107/Tourism/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path).squeeze() # .squeeze() to convert DataFrame to Series

# Correctly identify numeric and categorical features for the Tourism dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups'
]
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched',
    'Passport',
    'OwnCar'
]

# Ensure all numeric columns are actually numeric (handle potential string types from CSV)
for col in numeric_features:
    Xtrain[col] = pd.to_numeric(Xtrain[col], errors='coerce')
    Xtest[col] = pd.to_numeric(Xtest[col], errors='coerce')

# Impute missing numeric values with the median (simple strategy for demonstration)
for col in numeric_features:
    median_val = Xtrain[col].median()
    Xtrain[col].fillna(median_val, inplace=True)
    Xtest[col].fillna(median_val, inplace=True)

# Impute missing categorical values with the mode
for col in categorical_features:
    mode_val = Xtrain[col].mode()[0]
    Xtrain[col].fillna(mode_val, inplace=True)
    Xtest[col].fillna(mode_val, inplace=True)

# Class weight to handle imbalance for 'ProdTaken'
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid (adjusted for potentially smaller datasets/quicker run)
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.5, 0.7],
    'xgbclassifier__subsample': [0.6, 0.8]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='recall', n_jobs=-1) # Reduced CV for speed
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model
model_filename = "best_tourism_customer_predictor_v1.joblib"
joblib.dump(best_model, model_filename)

# Upload to Hugging Face (updated repo_id)
repo_id = "jit0107/tourism_customer_predictor"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the model space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=repo_type,
)
