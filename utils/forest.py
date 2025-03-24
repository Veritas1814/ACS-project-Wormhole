import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json

train_data = pd.read_csv("../data/iris_train.csv")
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train_encoded)

tree_structure = {
    "feature": [est.tree_.feature.tolist() for est in model.estimators_],
    "threshold": [est.tree_.threshold.tolist() for est in model.estimators_],
    "children_left": [est.tree_.children_left.tolist() for est in model.estimators_],
    "children_right": [est.tree_.children_right.tolist() for est in model.estimators_],
    "value": [est.tree_.value.tolist() for est in model.estimators_],
    "classes": label_encoder.classes_.tolist()
}

with open("forest.json", "w") as f:
    json.dump({"forest": tree_structure}, f, indent=4)

X_test = pd.read_csv("../data/iris_test.csv")
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
