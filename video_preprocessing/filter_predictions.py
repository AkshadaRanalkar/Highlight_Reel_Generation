import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

@dataclass
class FilterConfig:
    window_size: int
    method: str
    expansion: int

    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'method': self.method,
            'expansion': self.expansion
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FilterConfig':
        return cls(
            window_size=data['window_size'],
            method=data['method'],
            expansion=data['expansion']
        )

class PredictionProcessor:
    def __init__(self):
        self.param_options = {
            'window_size': [3, 5, 7, 9, 11],
            'method': ['moving_average', 'median', 'gaussian'],
            'expansion': [1, 2, 3, 4, 5]
        }

    def apply_moving_average(self, predictions, size):
        return np.convolve(predictions, np.ones(size) / size, mode='same')

    def apply_median_filter(self, predictions, size):
        return medfilt(predictions, kernel_size=size)

    def apply_gaussian_filter(self, predictions, sigma):
        return gaussian_filter1d(predictions, sigma=sigma)

    def expand_data(self, filtered_data, expansion):
        expanded = np.zeros_like(filtered_data)
        for i in range(len(filtered_data)):
            start = max(0, i - expansion)
            end = min(len(filtered_data), i + expansion + 1)
            expanded[start:end] = np.maximum(expanded[start:end], filtered_data[i])
        return expanded

    def process_predictions(self, predictions: List[int], params: FilterConfig) -> List[int]:
        predictions = np.array(predictions)
        if params.method == 'moving_average':
            filtered = self.apply_moving_average(predictions, params.window_size)
        elif params.method == 'median':
            filtered = self.apply_median_filter(predictions, params.window_size)
        elif params.method == 'gaussian':
            filtered = self.apply_gaussian_filter(predictions, params.window_size / 4)
        else:
            raise ValueError(f"Unknown filter method: {params.method}")

        expanded = self.expand_data(filtered, params.expansion)
        return (expanded > 0.5).astype(int).tolist()

    def calculate_metrics(self, predictions: List[int], actual: List[int]) -> Dict[str, float]:
        if len(predictions) != len(actual):
            raise ValueError("Predictions and actual values must have the same length.")

        pred_arr = np.array(predictions)
        actual_arr = np.array(actual)

        accuracy = np.mean(pred_arr == actual_arr)
        pred_changes = np.sum(np.abs(np.diff(pred_arr)))
        actual_changes = np.sum(np.abs(np.diff(actual_arr)))

        true_positives = np.sum((pred_arr == 1) & (actual_arr == 1))
        false_positives = np.sum((pred_arr == 1) & (actual_arr == 0))
        false_negatives = np.sum((pred_arr == 0) & (actual_arr == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1_score),
            'precision': float(precision),
            'recall': float(recall),
            'state_change_difference': float(abs(pred_changes - actual_changes))
        }

    def optimize_parameters(self, predictions: List[int], targets: List[int], param_grid: Optional[Dict] = None, metric: str = 'f1_score') -> Tuple[FilterConfig, Dict[str, float]]:
        if param_grid is None:
            param_grid = self.param_options

        best_params = None
        best_metrics = None
        highest_score = -float('inf')

        param_combinations = product(
            param_grid['window_size'],
            param_grid['method'],
            param_grid['expansion']
        )

        for window_size, method, expansion in param_combinations:
            params = FilterConfig(window_size, method, expansion)
            smoothed = self.process_predictions(predictions, params)
            metrics = self.calculate_metrics(smoothed, targets)

            score = metrics[metric]
            if score > highest_score:
                highest_score = score
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics

    def save_parameters(self, params: FilterConfig, filepath: str):
        with open(filepath, 'w') as file:
            json.dump(params.to_dict(), file, indent=2)

    def load_parameters(self, filepath: str) -> FilterConfig:
        with open(filepath, 'r') as file:
            return FilterConfig.from_dict(json.load(file))

def plot_results(raw_predictions: List[int], smoothed_predictions: List[int], targets: List[int], save_path: str):
    plt.figure(figsize=(15, 6))
    frames = range(len(raw_predictions))
    plt.step(frames, raw_predictions, where='post', label='Raw Predictions', alpha=0.7)
    plt.step(frames, smoothed_predictions, where='post', label='Smoothed Predictions', alpha=0.7)
    plt.step(frames, targets, where='post', label='Target Sequence', alpha=0.7)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.title('Comparison of Predictions')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Load input data
    predictions_data = pd.read_csv('logistic_predictions.csv')
    targets_data = pd.read_csv('target.csv')

    # Merge and sort datasets
    combined_data = pd.merge(predictions_data, targets_data, on='frame', suffixes=('_raw', '_target')).sort_values('frame')

    # Extract predictions and targets
    raw_predictions = combined_data['value_pred'].astype(int).tolist()
    target_values = combined_data['value_target'].astype(int).tolist()

    # Initialize processor and optimize parameters
    processor = PredictionProcessor()
    best_params, best_metrics = processor.optimize_parameters(raw_predictions, target_values, metric='f1_score')

    # Smooth predictions using optimal parameters
    smoothed_predictions = processor.process_predictions(raw_predictions, best_params)

    # Display best parameters and metrics
    print("\nOptimal Parameters:")
    print(json.dumps(best_params.to_dict(), indent=2))
    print("\nMetrics:")
    print(json.dumps(best_metrics, indent=2))

    # Save smoothed results to file
    results_df = pd.DataFrame({'frame': combined_data['frame'], 'smoothed_value': smoothed_predictions})
    results_df.to_csv('smoothed_predictions.csv', index=False)
    print("\nSmoothed predictions saved to 'smoothed_predictions.csv'.")

    # Plot results and save as image
    plot_results(raw_predictions, smoothed_predictions, target_values, 'predictions_comparison.png')
    print("Comparison plot saved to 'predictions_comparison.png'.")
