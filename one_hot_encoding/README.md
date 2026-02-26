# Car Price Prediction Using Linear Regression

## Project Description

In this project, I worked on predicting car selling prices using Linear Regression.  
The dataset contains prices of three different car models along with their mileage and age.

The goal was to:
1. Check if linear regression can be applied.
2. Predict prices for specific cars.
3. Calculate the model accuracy.

---

## Dataset

File used: `carprices.csv`

Columns:
- Car Model
- Mileage
- Age
- Sell Price

---

## Step 1: Data Visualization

I first plotted scatter plots to check the relationship between features and selling price.

```python
plt.scatter(df["Mileage"], df["Sell Price"])
plt.xlabel("Mileage")
plt.ylabel("Sell Price")
plt.show()

plt.scatter(df["Age"], df["Sell Price"])
plt.xlabel("Age")
plt.ylabel("Sell Price")
plt.show()
```

Observation:
- As mileage increases, sell price decreases.
- As age increases, sell price decreases.

This shows linear regression can be applied.

---

## Step 2: Convert Car Model to Numbers

Since machine learning models cannot handle text data, I converted car models into numeric values.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Car Model"] = le.fit_transform(df["Car Model"])
```

Then I applied One Hot Encoding to avoid treating car models as ordered numbers.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [("Car Model", OneHotEncoder(drop='first'), [0])],
    remainder="passthrough"
)

X = ct.fit_transform(X)
```

---

## Step 3: Train Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
```

---

## Step 4: Predictions

### 1) Price of Mercedes Benz (4 years old, 45000 mileage)

```python
model.predict([[0, 1, 45000, 4]])
```

### 2) Price of BMW X5 (7 years old, 86000 mileage)

```python
model.predict([[0, 0, 86000, 7]])
```

---

## Step 5: Model Score

```python
model.score(X, y)
```

The score shows how well the model fits the data.

---

## What I Learned

- How to visualize data before applying a model
- Why categorical variables must be converted to numeric
- How One Hot Encoding works
- How Linear Regression predicts values
- How to check model accuracy


## Conclusion

This project helped me understand how Linear Regression works with multiple variables and categorical data.  
The model successfully predicts car prices based on model type, mileage, and age.
