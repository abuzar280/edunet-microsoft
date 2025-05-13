import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

learners_df = pd.DataFrame({
    'LearnerID': range(1, 501),
    'Gender': np.random.choice(['Male', 'Female'], 500),
    'Age': np.random.randint(22, 50, 500),
    'Region': np.random.choice(['APAC', 'EMEA', 'AMER'], 500),
    'Role': np.random.choice(['Engineer', 'Manager', 'Analyst'], 500),
})

courses_df = pd.DataFrame({
    'CourseID': range(1, 11),
    'CourseName': [f'Course_{i}' for i in range(1, 11)],
    'Difficulty': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 10),
    'InstructorRating': np.random.uniform(3.0, 5.0, 10),
})

enrollment_df = pd.DataFrame({
    'LearnerID': np.random.choice(learners_df['LearnerID'], 1000),
    'CourseID': np.random.choice(courses_df['CourseID'], 1000),
    'EnrollmentDate': pd.to_datetime('2022-01-01') + pd.to_timedelta(np.random.randint(0, 365, 1000), unit='d'),
    'CompletionStatus': np.random.choice(['Completed', 'Dropped'], 1000, p=[0.7, 0.3])
})

performance_df = pd.DataFrame({
    'LearnerID': learners_df['LearnerID'],
    'Promotion': np.random.choice(['Yes', 'No'], 500, p=[0.2, 0.8]),
    'SalaryGrowth': np.random.uniform(0, 30, 500),
    'RoleChanges': np.random.choice([0, 1, 2], 500),
    'Retention': np.random.choice(['Yes', 'No'], 500, p=[0.85, 0.15])
})

impact_df = pd.merge(enrollment_df, learners_df, on='LearnerID', how='left')
impact_df = pd.merge(impact_df, courses_df, on='CourseID', how='left')
impact_df = pd.merge(impact_df, performance_df, on='LearnerID', how='left')

impact_df['Completed'] = (impact_df['CompletionStatus'] == 'Completed').astype(int)
impact_df['Promoted'] = (impact_df['Promotion'] == 'Yes').astype(int)
impact_df['HighSalaryGrowth'] = (impact_df['SalaryGrowth'] > 10).astype(int)
impact_df['Retained'] = (impact_df['Retention'] == 'Yes').astype(int)

np.random.seed(42)
impact_df['MarketDemandScore'] = np.random.uniform(0, 1, len(impact_df))

categorical_cols = ['Gender', 'Region', 'Role', 'Difficulty']
impact_df = pd.get_dummies(impact_df, columns=categorical_cols, drop_first=True)

excluded_cols = ['LearnerID', 'CourseID', 'EnrollmentDate', 'CompletionStatus', 'Promotion',
                 'SalaryGrowth', 'RoleChanges', 'Retention', 'Completed', 'Promoted',
                 'HighSalaryGrowth', 'Retained', 'CourseName']
features = [col for col in impact_df.columns if col not in excluded_cols]

X = impact_df[features]
y_completion = impact_df['Completed']
y_promotion = impact_df['Promoted']
y_salary = impact_df['HighSalaryGrowth']
y_retention = impact_df['Retained']

def train_and_evaluate(X, y, model, model_name):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
    print(f"{model_name} Mean CV AUC: {np.mean(cv_scores):.4f}")
    model.fit(X, y)
    return model

xgb_model_completion = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model_promotion = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
lr_model_salary = LogisticRegression(random_state=42, solver='liblinear')
rf_model_retention = RandomForestClassifier(random_state=42)

xgb_model_completion = train_and_evaluate(X, y_completion, xgb_model_completion, "XGBoost Completion")
xgb_model_promotion = train_and_evaluate(X, y_promotion, xgb_model_promotion, "XGBoost Promotion")
lr_model_salary = train_and_evaluate(X, y_salary, lr_model_salary, "Logistic Regression Salary Growth")
rf_model_retention = train_and_evaluate(X, y_retention, rf_model_retention, "Random Forest Retention")

gender_cols = [col for col in impact_df.columns if 'Gender_' in col]
for gender_col in gender_cols:
    idx = impact_df[gender_col] == 1
    gender_auc = roc_auc_score(y_promotion[idx], xgb_model_promotion.predict_proba(X[idx])[:, 1])
    print(f"Promotion AUC for {gender_col.replace('Gender_', '')}: {gender_auc:.4f}")

explainer = shap.TreeExplainer(xgb_model_promotion)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

impact_df['Promotion_Prob'] = xgb_model_promotion.predict_proba(X)[:, 1]
impact_df['Completion_Prob'] = xgb_model_completion.predict_proba(X)[:, 1]
impact_df['SalaryGrowth_Prob'] = lr_model_salary.predict_proba(X)[:, 1]
impact_df['Retention_Prob'] = rf_model_retention.predict_proba(X)[:, 1]

dashboard_df = impact_df[['LearnerID', 'Promotion_Prob', 'Completion_Prob', 
                          'SalaryGrowth_Prob', 'Retention_Prob', 'MarketDemandScore']]

print("\n📊 Dashboard-Ready Data Preview:")
print(dashboard_df.head())
