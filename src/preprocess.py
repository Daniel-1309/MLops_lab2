import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop null values and duplicates
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Encode sex, smoker, and region using one-hot encoding
    df = pd.get_dummies(df, columns=['region'])
    df['region_northwest'] = df['region_northwest'].map({True: 1, False: 0})
    df['region_southeast'] = df['region_southeast'].map({True: 1, False: 0})
    df['region_southwest'] = df['region_southwest'].map({True: 1, False: 0})
    df['region_northeast'] = df['region_northeast'].map({True: 1, False: 0})

    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    
    return df

if __name__ == "__main__":
    file_path = 'data/insurance.csv'  
    preprocessed_df = preprocess_data(file_path)
    preprocessed_df.to_csv('data/preprocessed_dataset.csv', index=False)  