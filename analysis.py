# ==========================================
# STUDENT ANALYTICS PROJECT - FINAL VERSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

print("====================================")
print("STATISTICAL DATA ANALYSIS PIPELINE")
print("====================================")

# -------------------------------
# 1. LOAD DATA
# -------------------------------

print("\n=== LOADING DATA ===")

# Make sure raw_data.csv is inside the same folder OR adjust path
df = pd.read_csv("data.csv")

print("\nBasic Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# 2. HANDLE MISSING VALUES
# -------------------------------

print("\n=== DATA CLEANING ===")

if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

print("Missing values handled successfully.")

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------

print("\n=== FEATURE ENGINEERING ===")

if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("Feature engineering completed.")

# -------------------------------
# 4. HYPOTHESIS TESTING
# -------------------------------

print("\n=== HYPOTHESIS TESTING ===")

if 'Survived' in df.columns and 'Sex' in df.columns:
    males = df[df['Sex'] == 0]['Survived']
    females = df[df['Sex'] == 1]['Survived']

    stat, p_value = ttest_ind(males, females)

    print("Two-Sample T-Test Results")
    print("Test Statistic:", stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant difference in survival.")
    else:
        print("Result: No statistically significant difference.")

# -------------------------------
# 5. VISUALIZATION
# -------------------------------

print("\n=== VISUALIZATION ===")

if 'Age' in df.columns:
    plt.figure()
    plt.hist(df['Age'], bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# 6. LOGISTIC REGRESSION
# -------------------------------

print("\n=== PREDICTIVE MODELING ===")

features = []

for col in ['Age', 'Sex', 'FamilySize']:
    if col in df.columns:
        features.append(col)

if 'Survived' in df.columns and len(features) > 0:

    X = df[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", auc)

    # Coefficient Interpretation
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })

    print("\nModel Coefficients:")
    print(coef_df.sort_values(by="Coefficient", ascending=False))

# -------------------------------
# 7. SQLITE DATABASE
# -------------------------------

print("\n=== SQLITE DATABASE INTEGRATION ===")

conn = sqlite3.connect("student_data.db")
df.to_sql("students", conn, if_exists="replace", index=False)

query = """
SELECT Sex,
       AVG(Survived) as survival_rate,
       COUNT(*) as total_passengers
FROM students
GROUP BY Sex
"""

result = pd.read_sql(query, conn)
print("\nSQL Query Result:")
print(result)

conn.close()

print("\n====================================")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("====================================")
