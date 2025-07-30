# Cross-Validation from Scratch

This project implements **K-Fold Cross-Validation** from scratch using only NumPy — no external ML libraries like scikit-learn. It demonstrates a core machine learning concept by manually splitting data, training models across folds, and averaging performance.

##  What It Does

- Splits dataset into `k` folds
- Trains a user-provided model on `k-1` folds
- Evaluates on the held-out fold
- Repeats for all folds and returns the average score

##  Skills Demonstrated

- Manual implementation of cross-validation logic
- Object-oriented programming with Python classes
- Use of NumPy for vectorized operations and data handling
- Random shuffling with reproducibility via seed

## ⚙️ How to Use

```python
from cv4 import CV

# Create your model class 
model = MyModel()

cv = CV(k=5, model=model, shuffle=True, random_seed=42)
cv.fit(X, y)
avg_score = cv.score()
