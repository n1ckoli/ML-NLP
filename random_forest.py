# Basic classification example using Scikit-learn:

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

cols = [dataset.data[:, i] for i in range(4)]

X = pd.DataFrame({k:v for k,v in zip(dataset.feature_names,cols)})
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)

pipe = Pipeline([("forest", RandomForestClassifier())])

params = {"forest__max_depth": [1, 2, 3]}

grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
model = grid.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy: ", accuracy_score(preds, y_test))

