# 1. Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib # For saving the model

# 2. Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = iris.target_names # ['setosa', 'versicolor', 'virginica']

# 3. Split data (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# The math: Logistic Regression finds the best line (or hyperplane)
# to separate the different flower classes.
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Save the trained model and the target names
joblib.dump(model, 'iris_model.joblib')
joblib.dump(target_names, 'iris_target_names.joblib')

print("Model trained and saved successfully!")