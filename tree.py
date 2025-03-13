import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import json

# Завантаження даних
train_data = pd.read_csv("iris_train.csv")
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

# Кодування міток
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Тренування моделі
model = DecisionTreeClassifier(max_depth = 20, random_state=42)
model.fit(X_train, y_train_encoded)

# Експорт параметрів дерева
tree_structure = {
    "feature": model.tree_.feature.tolist(),
    "threshold": model.tree_.threshold.tolist(),
    "children_left": model.tree_.children_left.tolist(),
    "children_right": model.tree_.children_right.tolist(),
    "value": model.tree_.value.tolist(),
    "classes": label_encoder.classes_.tolist()
}

# Збереження у JSON
with open("tree.json", "w") as f:
    json.dump({"tree": tree_structure}, f, indent=4)

print("Tree structure exported to tree.json")