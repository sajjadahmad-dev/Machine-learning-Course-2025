# Titanic Survival Prediction using Decision Tree

## Objective
Build a Decision Tree model to predict whether a passenger survived or not using the following features: Pclass, Sex, Age, and Fare.

Dataset:
https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv

## Steps Performed

1. Loaded the dataset using pandas.
2. Selected required columns: Survived, Pclass, Sex, Age, Fare.
3. Separated input features and target variable.
4. Converted categorical column (Sex) into numerical values using LabelEncoder.
5. Filled missing values in Age column using mean.
6. Split dataset into training and testing data (80% training, 20% testing).
7. Trained DecisionTreeClassifier model.
8. Calculated model accuracy using dt.score().
9. Predicted survival for a sample input.

## Result

The model was successfully trained and evaluated.  
The accuracy score shows how well the model predicts passenger survival.

This implementation demonstrates preprocessing, handling missing values, encoding categorical data, training a Decision Tree model, and evaluating performance.