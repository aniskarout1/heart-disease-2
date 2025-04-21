import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
df = pd.read_csv("heart_disease_data.csv")
x = df.drop(columns="target", axis=1)
y = df["target"]

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=40),
    "SVM": SVC(probability=True),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}")
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

# Save best model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the model name
with open("model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"Best Model: {best_model_name} with Accuracy: {best_score}")
