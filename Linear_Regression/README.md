# Linear Regression Practice Projects

In this repository, I worked on two regression exercises to understand how Linear Regression works in real-world problems.

---

# 1️⃣ Canada Per Capita Income Prediction

## Objective

Predict Canada's per capita income for the year 2020 using historical data.

Dataset used:
canada_per_capita_income.csv

---

## Step 1: Load and Visualize Data

```python
plt.scatter(df["year"], df["per capita income"], color="red", marker="+")
plt.xlabel("year")
plt.ylabel("per capita income")
plt.show()
```

Observation:
Income increases steadily over time.  
The data shows a linear trend, so Linear Regression can be applied.

---

## Step 2: Train Model

```python
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X, y)
```

---

## Step 3: Predict Income for 2020

```python
model.predict([[2020]])
```

The model predicts the per capita income for the year 2020.

---

## Model Parameters

```python
print(model.coef_)       # slope
print(model.intercept_)  # intercept
```

The model follows equation:

income = m × year + b

---

## Save and Load Model

Using Pickle:

```python
import pickle
with open("model_pickle","wb") as file:
    pickle.dump(model,file)
```

Using Joblib:

```python
import joblib
joblib.dump(model, "model.joblib")
```

This allows us to reuse the trained model without retraining.

---

# 2️⃣ Hiring Salary Prediction

## Objective

Build a model that helps HR decide salary based on:

- Experience
- Test Score
- Interview Score

Dataset used:
hiring.csv

---

## Step 1: Data Cleaning

Fill missing experience with "zero":

```python
df["experience"] = df["experience"].fillna("zero")
```

Convert words to numbers:

```python
from word2number import w2n
df["experience"] = df["experience"].apply(w2n.word_to_num)
```

Fill missing test scores with median:

```python
df["test_score"] = df["test_score"].fillna(df["test_score"].median())
```

---

## Step 2: Train Model

```python
model = linear_model.LinearRegression()
model.fit(df[["experience","test_score","interview_score"]], df[["salary"]])
```

---

## Step 3: Predictions

### Candidate 1:
2 years experience, 9 test score, 6 interview score

```python
model.predict([[2,9,6]])
```

### Candidate 2:
12 years experience, 10 test score, 10 interview score

```python
model.predict([[12,10,10]])
```

---

## Model Details

```python
model.coef_
model.intercept_
```

The model calculates salary using:

salary = b + (m1 × experience) + (m2 × test_score) + (m3 × interview_score)

---

# What I Learned

- How Linear Regression works with single and multiple variables
- How to clean missing data
- How to convert text data into numeric format
- How to save and reuse trained models
- How to interpret model coefficients

