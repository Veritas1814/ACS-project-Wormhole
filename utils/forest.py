import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import os
import sys

# Get file paths from command line arguments
if len(sys.argv) != 5:
    print("Usage: python script.py <train_file> <test_file> <forest_output_file> <predictions_output_file>")
    sys.exit(1)

# Define data directory

data_dir = "data"


# Get file paths from arguments
train_file = os.path.join(data_dir, sys.argv[1])
test_file = os.path.join(data_dir, sys.argv[2])
forest_output_file = os.path.join(data_dir, sys.argv[3])
predictions_output_file = os.path.join(data_dir, sys.argv[4])

# Read the training data
train_data = pd.read_csv(train_file)
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=5000, random_state=42)
model.fit(X_train, y_train_encoded)

# Save Random Forest structure to JSON
forest_structure = {
    "feature": [est.tree_.feature.tolist() for est in model.estimators_],
    "threshold": [est.tree_.threshold.tolist() for est in model.estimators_],
    "children_left": [est.tree_.children_left.tolist() for est in model.estimators_],
    "children_right": [est.tree_.children_right.tolist() for est in model.estimators_],
    "value": [est.tree_.value.tolist() for est in model.estimators_],
    "classes": label_encoder.classes_.tolist()
}

with open(forest_output_file, "w") as f:
    json.dump({"forest": forest_structure}, f, indent=4)

# Read the test data and make predictions
X_test = pd.read_csv(test_file)
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Save the predictions to a CSV file
predictions = pd.DataFrame({"Predicted_Species": y_pred})
predictions.to_csv(predictions_output_file, index=False)

print(f"Forest structure saved to {forest_output_file}")
print(f"Predictions saved to {predictions_output_file}")
