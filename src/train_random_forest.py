import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load training data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

y_train = y_train.values.ravel()

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/rndm_frst_model.pkl")