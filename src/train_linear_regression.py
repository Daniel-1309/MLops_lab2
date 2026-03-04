import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load training data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

y_train = y_train.values.ravel()

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/linear_regression_model.pkl')