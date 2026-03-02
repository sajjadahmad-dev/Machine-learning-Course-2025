# Model Comparison using Cross Validation on Iris Dataset

## Objective

Use the Iris flower dataset from `sklearn` and evaluate the performance of different machine learning models using 5-fold cross validation.

Models evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## Dataset

Dataset: `sklearn.datasets.load_iris`

- 150 samples
- 4 features
- 3 classes (Setosa, Versicolor, Virginica)

---

## Method

- Loaded Iris dataset.
- Extracted features (X) and target (y).
- Used `cross_val_score()` with `cv=5`.
- Calculated average accuracy for each model.

---

## Results (Approximate)

| Model | Average Accuracy |
|--------|------------------|
| Logistic Regression | ~0.96–0.98 |
| SVM | ~0.97–0.99 |
| Decision Tree | ~0.94–0.96 |
| Random Forest | ~0.96–0.99 |

---

## Conclusion

Using 5-fold cross validation, **SVM and Random Forest** achieved the highest performance on the Iris dataset.

Cross validation provides a more reliable performance estimate compared to a single train-test split.