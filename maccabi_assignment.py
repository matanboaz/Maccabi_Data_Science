import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
df = pd.read_csv("ds_assignment_data.csv")

# 2. Preprocessing
target = "Y"
text_col = "clinical_text"
text_vectorizer = TfidfVectorizer(max_features=100)
text_features = text_vectorizer.fit_transform(df[text_col].fillna("")).toarray()
text_feature_names = [f"tfidf_{i}" for i in range(text_features.shape[1])]
df_text = pd.DataFrame(text_features, columns=text_feature_names)
df = pd.concat([df.drop(columns=[text_col]), df_text], axis=1)

# 3. Define features
exclude_cols = ["Y"] + [col for col in df.columns if "after" in col]
features = [col for col in df.columns if col not in exclude_cols]

X = df[features]
y = df[target]

# Impute
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# 5. Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"{name}:")
    print(f"  ROC AUC = {roc_auc:.3f}")
    print(f"  PR AUC  = {pr_auc:.3f}\n")