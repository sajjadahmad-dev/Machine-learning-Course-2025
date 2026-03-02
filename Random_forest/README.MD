# Random Forest Classifier on Iris Dataset

## Objective

Use the famous Iris flower dataset from `sklearn.datasets` to:

- Predict flower species using Random Forest Classifier
- Measure prediction score using default number of trees
- Fine-tune the model by changing the number of trees (`n_estimators`)
- Find the best accuracy

---

## Dataset

Dataset used: `sklearn.datasets.load_iris`

- Total samples: 150
- Features: 4
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Classes: 3
  - Setosa
  - Versicolor
  - Virginica

---

## Steps Performed

1. Loaded the Iris dataset.
2. Converted dataset into a Pandas DataFrame.
3. Added target column.
4. Split dataset into:
   - 80% training data
   - 20% testing data
5. Trained Random Forest Classifier using default parameters.
6. Measured model accuracy.
7. Tuned the model by changing number of trees (`n_estimators`).
8. Evaluated model performance using Confusion Matrix.

---

## Model Training

### Default Model

- Used default `RandomForestClassifier()`
- Measured prediction accuracy on test set.

### Fine Tuning

Tested different numbers of trees:

- 5 trees
- 10 trees
- 20 trees
- 50 trees
- 100 trees
- 200 trees

Observed accuracy for each configuration.

---

## Results

- Default accuracy: Approximately 0.93 – 1.0
- Best accuracy achieved: 1.0 (100%)
- Increasing number of trees improves stability.
- After certain number (50–100), accuracy stabilizes.

---

## Confusion Matrix

Generated confusion matrix to evaluate classification performance.

- Diagonal values represent correct predictions.
- Off-diagonal values represent misclassifications.

Model correctly classified almost all test samples.

---

## Conclusion

Random Forest performs extremely well on the Iris dataset due to clear class separation.

Increasing the number of trees improves model stability but does not significantly improve accuracy after a certain point.

The tuned Random Forest model achieved perfect classification accuracy on the test set.