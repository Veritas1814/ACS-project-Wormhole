import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import json

train_data = pd.read_csv("iris_train.csv")
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train_encoded)

tree_structure = {
    "feature": model.tree_.feature.tolist(),
    "threshold": model.tree_.threshold.tolist(),
    "children_left": model.tree_.children_left.tolist(),
    "children_right": model.tree_.children_right.tolist(),
    "value": model.tree_.value.tolist(),
    "classes": label_encoder.classes_.tolist()
}

with open("tree.json", "w") as f:
    json.dump({"tree": tree_structure}, f, indent=4)

X_test = pd.read_csv("iris_test.csv")
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)