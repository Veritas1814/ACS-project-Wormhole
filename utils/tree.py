import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import json
import os
import sys

# Get file paths from command line arguments
if len(sys.argv) != 5:
    print("Usage: python script.py <train_file> <test_file> <tree_output_file> <predictions_output_file>")
    sys.exit(1)

# Define data directory

data_dir = "data"


# Get file paths from arguments
train_file = os.path.join(data_dir, sys.argv[1])
test_file = os.path.join(data_dir, sys.argv[2])
tree_output_file = os.path.join(data_dir, sys.argv[3])
predictions_output_file = os.path.join(data_dir, sys.argv[4])

# Read the training data
train_data = pd.read_csv(train_file)
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train_encoded)

# Save tree structure to JSON
tree_structure = {
    "feature": model.tree_.feature.tolist(),
    "threshold": model.tree_.threshold.tolist(),
    "children_left": model.tree_.children_left.tolist(),
    "children_right": model.tree_.children_right.tolist(),
    "value": model.tree_.value.tolist(),
    "classes": label_encoder.classes_.tolist()
}

with open(tree_output_file, "w") as f:
    json.dump({"tree": tree_structure}, f, indent=4)

# Read the test data and make predictions
X_test = pd.read_csv(test_file)
y_pred_encoded = model.predict(X_test)
predictions = pd.DataFrame({"Predicted_Species": y_pred_encoded})

predictions.to_csv(predictions_output_file, index=False)

print(f"Tree saved to {tree_output_file}")
print(f"Predictions saved to {predictions_output_file}")
