# Support Vector Machine (SVM) on Digits Dataset

## Objective

Train an SVM classifier using the sklearn `load_digits` dataset and:

- Compare accuracy using **RBF kernel** and **Linear kernel**
- Tune **C (regularization)** and **gamma** parameters
- Use **80% training data** and **20% testing data**
- Achieve the highest possible accuracy

---

## Dataset

Dataset used: `sklearn.datasets.load_digits`

- 1797 samples
- 64 features (8x8 pixel images)
- 10 target classes (digits 0–9)

---

## Steps Performed

1. Imported required libraries.
2. Loaded digits dataset.
3. Converted dataset into pandas DataFrame.
4. Separated features (X) and target (y).
5. Split dataset into 80% training and 20% testing.
6. Applied **StandardScaler** for feature scaling (important for SVM).
7. Trained SVM using:
   - RBF kernel
   - Linear kernel
8. Tuned model using different values of:
   - C (regularization parameter)
   - gamma (kernel coefficient)
9. Compared accuracy scores.

---

## Results

- Linear Kernel Accuracy: ~97–98%
- RBF Kernel Accuracy (after scaling): ~98%
- Tuned RBF Model Accuracy: ~98–99%

The tuned RBF kernel achieved the highest performance.

---

## Conclusion

SVM performs very well on the digits dataset.  
Feature scaling significantly improves performance for RBF kernel.  
Proper tuning of C and gamma helps achieve better generalization and higher accuracy.