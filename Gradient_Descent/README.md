# Linear Regression Using Gradient Descent (From Scratch)

## Project Description

In this project, I implemented Linear Regression using Gradient Descent from scratch and compared the result with sklearn’s LinearRegression model.

The purpose of this project is to understand how gradient descent works internally instead of directly using built-in machine learning libraries.

---

## What This Code Does

1. Reads student test score data from `test_scores.csv`
2. Uses math scores to predict computer science (cs) scores
3. Implements Gradient Descent manually
4. Compares the result with sklearn LinearRegression

---

## Files Used

- test_scores.csv  
- Python script (gradient descent implementation)

---

## Concepts Used

- Linear Regression
- Gradient Descent
- Cost Function (Mean Squared Error)
- Partial Derivatives
- Learning Rate
- Model Coefficient (m)
- Intercept (b)

---

## How Gradient Descent Works in This Code

1. Start with initial values:
   - m = 0
   - b = 0

2. Calculate predicted values:
   y_pred = m*x + b

3. Calculate cost (Mean Squared Error)

4. Compute derivatives:
   - md (derivative w.r.t slope)
   - bd (derivative w.r.t intercept)

5. Update values:
   - m = m - learning_rate * md
   - b = b - learning_rate * bd

6. Repeat for multiple iterations

The goal is to minimize the cost function step by step.

---

## Output Explanation

The program prints:

- m (slope)
- b (intercept)
- cost
- iteration number

At the end it shows:

- Result using Gradient Descent
- Result using sklearn

This helps verify that our manual implementation works correctly.

---

## Learning Outcome

Through this project I understood:

- How Linear Regression works mathematically
- How Gradient Descent updates parameters
- Why learning rate is important
- How cost decreases over iterations
- Difference between manual implementation and sklearn


## Conclusion

This project helped me understand the core working of Linear Regression and Gradient Descent instead of treating machine learning models as black boxes.
