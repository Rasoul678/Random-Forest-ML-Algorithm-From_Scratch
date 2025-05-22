from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForestClassifier

# Load data
data = load_wine()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
n_trees = 100
clf = RandomForestClassifier(n_trees=n_trees, max_depth=5, n_features=2)
print(f"Running Random Forest with {n_trees} trees ...")
clf.fit(X_train, y_train)

# Predict and evaluate
y_predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predictions))
