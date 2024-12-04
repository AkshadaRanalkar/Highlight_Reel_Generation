import pandas as pd
import matplotlib.pyplot as plt

def load_data(raw_file, smoothed_file, target_file):
    # Load the data from CSV files
    raw_predictions = pd.read_csv(raw_file)
    smoothed_predictions = pd.read_csv(smoothed_file)
    target_data = pd.read_csv(target_file)
    
    # Ensure the frame columns are integers for proper merging
    raw_predictions['frame'] = raw_predictions['frame'].astype(int)
    smoothed_predictions['frame'] = smoothed_predictions['frame'].astype(int)
    target_data['frame'] = target_data['frame'].astype(int)

    # Merge the data on the 'frame' column
    data = raw_predictions.merge(smoothed_predictions, on='frame', suffixes=('_raw', '_smooth'))
    data = data.merge(target_data, on='frame')

    return data

def plot_data(data):
    plt.figure(figsize=(15, 6))
    plt.plot(data['frame'], data['value_raw'], label='Raw Predictions', linestyle='-', marker='', color='orange', alpha=0.75)
    plt.plot(data['frame'], data['value_smooth'], label='Smoothed Predictions', linestyle='-', marker='', color='green', alpha=0.75)
    plt.plot(data['frame'], data['value'], label='Target Sequence', linestyle='-', marker='', color='blue', alpha=0.75)
    
    plt.title('Comparison of Raw and Smoothed Predictions')
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # File paths
    raw_file = 'predictions.csv'
    smoothed_file = 'smoothed_predictions.csv'
    target_file = 'target.csv'
    
    # Load data
    data = load_data(raw_file, smoothed_file, target_file)
    
    # Plot data
    plot_data(data)

if __name__ == "__main__":
    main()
