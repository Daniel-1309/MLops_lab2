import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

y_test = y_test.values.ravel()

# Load model
model = joblib.load("model/rndm_frst_model.pkl")

# Predictions
predictions = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
accuracy = model.score(X_test, y_test)

metrics = {
    "mse": mse,
    "r2": r2, 
    "accuracy in %": accuracy*100
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f)

# Plot
plt.scatter(y_test, predictions)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted")

plt.savefig("reports/prediction_plot.png")