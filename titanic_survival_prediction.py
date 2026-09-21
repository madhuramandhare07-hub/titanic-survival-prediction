"""
Titanic Survival Prediction - Beginner ML Project
====================================================
100% free tools: pandas, seaborn, scikit-learn
No API keys, no paid services, no Kaggle account required.

Run with: python titanic_survival_prediction.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# STEP 1: Load the data (free, built into seaborn)
# ---------------------------------------------------------
df = sns.load_dataset('titanic')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------
# STEP 2: Clean the data
# ---------------------------------------------------------
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'deck']]

# Fill missing numeric values with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill missing categorical values with the most common value
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Encode categorical columns as numbers
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------
# STEP 2.5: Feature engineering (new, improved features)
# ---------------------------------------------------------
# Total family members aboard, including the passenger themself
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Was this passenger traveling alone?
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Did this passenger have a recorded cabin? (proxy for wealth/deck location)
df['has_cabin'] = df['deck'].notna().astype(int)
df = df.drop('deck', axis=1)

print("\nEngineered features added: family_size, is_alone, has_cabin")

# ---------------------------------------------------------
# STEP 3: Split into train/test sets
# ---------------------------------------------------------
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# STEP 4: Train models
# ---------------------------------------------------------
# Model 1: Logistic Regression (simple baseline)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_preds)

# Model 2: Random Forest (usually stronger)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"\nLogistic Regression Accuracy: {log_reg_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ---------------------------------------------------------
# STEP 4.5: Hyperparameter tuning with GridSearchCV
# ---------------------------------------------------------
# GridSearchCV tests many combinations of settings and uses cross-validation
# (splitting the training data 5 different ways) to find the most reliable one.
# This is more trustworthy than judging a model on a single train/test split.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
tuned_preds = best_rf.predict(X_test)
tuned_acc = accuracy_score(y_test, tuned_preds)

print(f"\nBest hyperparameters found: {grid_search.best_params_}")
print(f"Cross-validation accuracy (5-fold): {grid_search.best_score_:.4f}")
print(f"Tuned model accuracy on test set: {tuned_acc:.4f}")
print("\nNote: cross-validation accuracy is the more reliable estimate on small")
print("datasets like this one (179 test samples) since a single test split can")
print("vary a lot by chance.")

# Use the tuned model going forward
rf = best_rf
rf_preds = tuned_preds

# ---------------------------------------------------------
# STEP 5: Evaluate the best model in detail
# ---------------------------------------------------------
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# ---------------------------------------------------------
# STEP 6: Feature importance (what mattered most)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

# Plot and save feature importance chart
plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='steelblue')
plt.title('Feature Importance - Titanic Survival Prediction')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nSaved chart: feature_importance.png")

# ---------------------------------------------------------
# STEP 7: Try a prediction on a custom passenger
# ---------------------------------------------------------
# Example: 28-year-old woman, 1st class, no siblings/spouse, no parents/children,
# paid $80 fare, embarked from Cherbourg
sample_passenger = pd.DataFrame([{
    'pclass': 1,
    'sex': 1,       # female
    'age': 28,
    'sibsp': 0,
    'parch': 0,
    'fare': 80,
    'embarked': 1,  # Cherbourg
    'family_size': 1,
    'is_alone': 1,
    'has_cabin': 1
}])

prediction = rf.predict(sample_passenger)[0]
probability = rf.predict_proba(sample_passenger)[0][1]

print(f"\nSample passenger survival prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
print(f"Survival probability: {probability:.2%}")
