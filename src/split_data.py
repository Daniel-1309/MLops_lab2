# Split data into training and testing sets
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path, test_size=0.2, random_state=42):
    # Load the preprocessed dataset
    df = pd.read_csv(file_path)

    # Separate features and target variable
    X = df.drop('charges', axis=1)  # Assuming 'charges' is the target variable
    y = df['charges']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save the training and testing sets to CSV files
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = 'data/preprocessed_dataset.csv'  # Update with your preprocessed dataset path
    split_data(file_path)