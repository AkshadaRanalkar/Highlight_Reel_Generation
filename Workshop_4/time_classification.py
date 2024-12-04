import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_and_evaluate(data_path, target_path):
    # Load provided_data.csv
    data = pd.read_csv(data_path, header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
    # Convert 'effort' column to numeric and impute missing values
    data['effort'] = pd.to_numeric(data['effort'], errors='coerce').interpolate(method='linear')

    # Load target.csv and ensure data types are consistent for merging
    target = pd.read_csv(target_path)
    target['frame'] = target['frame'].astype(int)

    # Merge data and target on 'frame'
    merged = pd.merge(data, target, on='frame', how='inner')

    # Select features and target variable
    features = ['xc', 'yc', 'w', 'h', 'effort']
    X = merged[features]
    y = merged['value']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train Logistic Regression Model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Output the classification report
    print(classification_report(y_test, y_pred))

    # Optional: Save the predictions with frame numbers for further analysis
    predictions_df = pd.DataFrame({'frame': merged.loc[y_test.index, 'frame'], 'predicted_value': y_pred})
    predictions_df.to_csv('logistic_predictions.csv', index=False)
    print("Predictions saved to logistic_predictions.csv")

# Example usage, ensure your paths are correct
train_and_evaluate('provided_data.csv', 'target.csv')
