# Salary Prediction using Multiple Linear Regression

A beginner-level machine learning project that predicts employee salary from years of experience, test score and department using Multiple Linear Regression (scikit-learn).

## Dataset
The dataset is synthetic. If `salary_data.csv` is not found, the script generates 500 rows using a fixed random seed, with a known salary formula plus random noise. Because the data is synthetic, the model scores are high by design and do not represent real-world performance.

## Features
- `experience` (years), `test_score`, `department` (IT, HR, Sales)

Target: `salary`

## What the script does
1. Loads or generates the dataset
2. Converts the categorical `department` column into numbers using one-hot encoding
3. Compares a single-feature model (experience only) with a multiple-feature model
4. Evaluates both using RMSE and R2 score
5. Saves the better model as `salary_model.pkl`
6. Predicts the salary for an example employee

## How to run
pip install -r requirements.txt
python salary_prediction.py

## Results 
Single feature (experience): RMSE (RMSE 4,012.35), R2 (0.9871)
Multiple features: RMSE (add value), R2 (add value)

## Tech stack
Python, pandas, NumPy, scikit-learn, joblib

## Possible improvements
- Use a real-world dataset
- Select the model with cross-validation instead of a single test split
- Use a scikit-learn Pipeline with OneHotEncoder
