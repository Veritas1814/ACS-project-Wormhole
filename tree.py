import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load training data
train_data = pd.read_csv("iris_train.csv")
X_train = train_data.drop(columns=["Species"])
y_train = train_data["Species"]

# Encode target labels
y_encoder = LabelEncoder()
y_train_encoded = y_encoder.fit_transform(y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_encoded)

# Load test data
test_data = pd.read_csv("iris_test.csv")

# Predict
predictions_encoded = model.predict(test_data)
predictions = y_encoder.inverse_transform(predictions_encoded)

# Save output
output = test_data.copy()
output["Species"] = predictions
output.to_csv("output_py.csv", index=False)
