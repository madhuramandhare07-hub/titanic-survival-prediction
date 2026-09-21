# Titanic Survival Prediction

A beginner machine learning project predicting passenger survival on the Titanic, using only free, open-source tools.

## Tools used (all free)
- Python
- pandas (data cleaning)
- seaborn (dataset + visualization)
- scikit-learn (machine learning models)
- matplotlib (charts)

## How to run
1. Install dependencies: `pip install pandas seaborn scikit-learn matplotlib`
2. Run: `python titanic_survival_prediction.py`

## What it does
1. Loads the classic Titanic dataset (891 passengers, built into seaborn — no download needed)
2. Cleans missing data (age, embarked port) and encodes categorical features
3. Engineers new features: family size, whether the passenger was alone, and whether they had a recorded cabin
4. Trains Logistic Regression and Random Forest, then tunes Random Forest with GridSearchCV (5-fold cross-validation)
5. Shows which features mattered most for survival
6. Predicts survival for a custom sample passenger

## Results
- **Logistic Regression accuracy:** ~80%
- **Random Forest (tuned) cross-validation accuracy:** ~83%
- **Most important feature by far:** sex (43% importance), followed by fare, class, and age

### A real lesson learned in this project
Adding features and tuning hyperparameters didn't automatically improve accuracy on the single 179-passenger test split — it actually looked slightly lower there. But the 5-fold **cross-validation** score (which tests across 5 different data splits) did improve, and is the more statistically reliable number on a dataset this small. This is a genuinely useful thing to mention in an interview: single train/test splits can be misleading on small datasets, so cross-validation gives a more trustworthy estimate of how a model will generalize.

## Resume bullet point (example)
> Built and tuned a machine learning classifier in Python (pandas, scikit-learn) to predict Titanic passenger survival, engineering new features and using GridSearchCV with 5-fold cross-validation to achieve 83% validated accuracy; identified sex, fare, and passenger class as the strongest survival predictors.

## Next steps to extend this project
- Try XGBoost or LightGBM instead of Random Forest (also free, often stronger)
- Get the full dataset (with passenger names) from Kaggle to extract title (Mr/Mrs/Miss) as a feature
- Deploy it as a simple web app using Streamlit (also free)
- Push the code to GitHub to link on your resume
