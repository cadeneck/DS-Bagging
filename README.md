
# Ensemble Learning and Cross-Validation Implementation

This repository contains Python implementations of Bagging, Cross-Validation, and OneHotEncoder. The implementations leverage popular libraries such as NumPy, SciPy, and Matplotlib for data manipulation and visualization. 

## Bagging
The Bagging (Bootstrap Aggregating) implementation uses the DecisionTreeClassifier from Scikit-Learn as the base learner. It creates an ensemble of decision tree classifiers and trains each tree on a bootstrap sample of the input data. Bagging aims to reduce variance and avoid overfitting. 

### Features:
- Bagging classifier with customizable number of learners.
- Utilizes DecisionTreeClassifier as the base learner.
- Supports OneHot encoding of categorical data for model training.

## Cross-Validation
The Cross-Validation module implements k-fold cross-validation to evaluate the performance of machine learning models. It splits the dataset into k folds, trains the model on k-1 folds, and validates it on the remaining fold. This process repeats k times, with each fold used exactly once as the validation data.

### Features:
- K-fold cross-validation with customizable number of folds.
- Splitting of data into training and validation sets.
- Calculation of average accuracy across all folds.

## OneHotEncoder
The OneHotEncoder class encodes categorical variables as binary vectors, with each unique category represented by a unique position in the vector. This encoding is necessary for models that require numerical input data.

### Features:
- Conversion of categorical data into one-hot encoded format.
- Fit method to learn the categories from the input data.
- Encode method to transform data based on learned categories.

## Usage
To use these implementations, simply import the necessary classes from their respective modules. Below is an example of how to use the Bagging classifier with cross-validation:

```python
from bagging import Bagging
from cross_validation import CrossValidation
from utils import load_data

X, y = load_data('your_dataset.csv')
cv = CrossValidation(k=5, random_seed=42)

best_model, best_param, scores = cv.get_best_model(X, y, params=[5, 10, 15])
print(f"Best number of learners: {best_param}")
```

## Dependencies
- NumPy
- Scikit-Learn
- Matplotlib
- tqdm (for progress bars)

