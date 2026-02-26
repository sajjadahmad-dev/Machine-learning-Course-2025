# Employee Retention Prediction using Logistic Regression

## Project Overview
This project analyzes employee retention using the HR Analytics dataset from Kaggle. The objective is to identify key factors affecting employee attrition and build a Logistic Regression model to predict whether an employee will leave the company.

Dataset Link:
https://www.kaggle.com/giripujar/hr-analytics

---

## Step 1: Load Dataset

```python
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("HR_comma_sep.csv")
print(df.head())
```

Target Column:
- left = 0 → Employee stayed
- left = 1 → Employee left

---

## Step 2: Exploratory Data Analysis (EDA)

Separate employees who left vs retained:

```python
left = df[df.left == 1]
retained = df[df.left == 0]

print(left.shape)
print(retained.shape)
```

Compare averages:

```python
df.groupby('left').mean(numeric_only=True)
```

### Key Findings:
- Employees who left had lower satisfaction levels.
- Employees who left worked more monthly hours.
- Employees who received promotions were more likely to stay.
- Employees with low salary left more frequently.

Selected Important Features:
- satisfaction_level
- average_montly_hours
- promotion_last_5years
- salary

---

## Step 3: Salary vs Retention (Bar Chart)

```python
pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.title("Salary vs Employee Retention")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.show()
```

Result: Employees with higher salaries are less likely to leave.

---

## Step 4: Department vs Retention (Bar Chart)

```python
pd.crosstab(df.Department, df.left).plot(kind='bar')
plt.title("Department vs Employee Retention")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.show()
```

Result: Department has some impact but not strong enough to include in the model.

---

## Step 5: Convert Salary to Dummy Variables

Machine learning models cannot process text data, so we convert salary categories into numeric values using one-hot encoding.

```python
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)
```

Now all features are numeric.

---

## Step 6: Build Logistic Regression Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df_with_dummies
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

## Step 7: Measure Accuracy

```python
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
```

Model Accuracy: ~78%

---

## Conclusion

- Satisfaction level is the strongest predictor of employee retention.
- Low salary increases probability of leaving.
- Promotions improve employee retention.
- Logistic Regression successfully predicts employee attrition with approximately 78% accuracy.

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn