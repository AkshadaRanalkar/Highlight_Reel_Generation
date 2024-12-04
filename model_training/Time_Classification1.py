import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ray
import warnings

#Load and Prepare Data
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

#Feature Scaling mean 0 and standard deviation 1
# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort']
X = merged[features].astype(np.float32)  # Reduce memory usage by using float32
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Define Helper Functions
# Function to create lag features for time series data (optimized for memory)
def create_lag_features(X, window_size):
    lagged_features = []
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        lagged_features.append(X_shifted)
    X_lagged = pd.concat(lagged_features, axis=1)
    return X_lagged.dropna()

#Define Model Training Function
@ray.remote
def create_model(X, y, window_size, n_estimators, learning_rate, max_depth):
    print(f"Received X type inside worker: {type(X)}")  # Debugging line
    
    X_lagged = create_lag_features(X, window_size)
    y_lagged = y.iloc[window_size - 1:]

    # Align indices
    y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
    X_lagged = X_lagged.reset_index(drop=True)
    
    # Split into train and test sets
    split_index = int(len(X_lagged) * 0.7)
    X_train = X_lagged.iloc[:split_index]
    X_test = X_lagged.iloc[split_index:]
    y_train = y_lagged.iloc[:split_index]
    y_test = y_lagged.iloc[split_index:]
    
    # Train and evaluate the model
    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, 
                        random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    return window_size, n_estimators, learning_rate, max_depth, clf.score(X_test, y_test)

#Define Hyperparameter Grid and Initialize Ray
# Define parameter grid (reduced size to avoid excessive memory usage)
param_grid = {
    'window_size': [1, 2, 5, 10],  # Reduced range for testing
    'n_estimators': [50, 100],  # Reduced range for testing
    'learning_rate': [0.01, 0.1],  # Reduced range for testing
    'max_depth': [3, 5]
}

# Initialize Ray
ray.init(ignore_reinit_error=True)

#Run Grid Search in Parallel
# Perform grid search
results = []
total_iterations = len(param_grid['window_size']) * len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth'])
futures = []

# Put X_scaled directly into Ray's object store
X_ref = ray.put(X_scaled)
print(f"X_ref type before passing to workers: {type(X_ref)}")  # Debugging line

for window_size in param_grid['window_size']:
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                futures.append(create_model.remote(X_ref, y, window_size, n_estimators, learning_rate, max_depth))

with tqdm(total=total_iterations, desc="Parameter Search") as pbar:
    while futures:
        done, futures = ray.wait(futures)
        results.extend(ray.get(done))
        pbar.update(len(done))

# Shut down Ray
ray.shutdown()

#Analyze Results and Plot Performance
# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['window_size', 'n_estimators', 'learning_rate', 'max_depth', 'score'])

# Create heatmap (for a specific hyperparameter combination visualization)
plt.figure(figsize=(12, 8))
pivot_table = results_df.pivot_table(index='window_size', columns='n_estimators', values='score', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Model Performance: Window Size vs. Number of Estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Window Size')
plt.tight_layout()
plt.savefig('Parameter_Search_Heatmap.png')
plt.close()

#Train Final Model with Best Parameters
# Find best parameters
best_result = results_df.loc[results_df['score'].idxmax()]
print(f"Best parameters: Window Size = {best_result['window_size']}, "
      f"N Estimators = {best_result['n_estimators']}, Learning Rate = {best_result['learning_rate']}, Max Depth = {best_result['max_depth']}")
print(f"Best score: {best_result['score']:.3f}")

# Train final model with best parameters
best_window_size = int(best_result['window_size'])
best_n_estimators = int(best_result['n_estimators'])
best_learning_rate = best_result['learning_rate']
best_max_depth = int(best_result['max_depth'])

X_lagged = create_lag_features(X_scaled, best_window_size)
y_lagged = y.iloc[best_window_size - 1:]
frames_lagged = merged['frame'].iloc[best_window_size - 1:]

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]

# Train final model
clf = XGBClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate, max_depth=best_max_depth, 
                    random_state=42, use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)

#Evaluate Final Model
# Predict on the test set
y_pred = clf.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(clf, 'final_xgb_model.pkl')