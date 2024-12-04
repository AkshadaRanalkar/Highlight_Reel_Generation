import numpy as np
import pandas as pd

class PredictionSmoother:
    def __init__(self, window_size=5, min_duration=5, hysteresis=0.6):
        self.window_size = window_size
        self.min_duration = min_duration
        self.hysteresis = hysteresis

    def smooth_predictions(self, predictions):
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - self.window_size // 2)
            end = min(len(predictions), i + self.window_size // 2 + 1)
            window = predictions[start:end]
            active_ratio = np.mean(window)

            threshold = self.hysteresis if smoothed and smoothed[-1] else 1 - self.hysteresis
            smoothed.append(1 if active_ratio >= threshold else 0)

        return smoothed

def apply_smoothing(file_path):
    df = pd.read_csv(file_path)
    smoother = PredictionSmoother()
    smoothed_values = smoother.smooth_predictions(df['value'].tolist())

    df['smoothed_value'] = smoothed_values
    df.to_csv('smoothed_predictions.csv', index=False)
    print("Smoothed predictions saved to 'smoothed_predictions.csv'")

# Example usage
apply_smoothing('predictions.csv')
