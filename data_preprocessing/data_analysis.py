import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def data_analysis(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['frame', 'x_coord', 'y_coord', 'feature1', 'feature2', 'optional_feature']

    derivative = np.diff(df['x_coord'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(df['frame'], df['x_coord'])
    ax1.set_ylabel('X Coordinate')
    ax1.set_title('X Coordinate vs Frame Number')
    ax1.grid(True)

    ax2.plot(df['frame'][1:], derivative)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Derivative')
    ax2.set_title('Derivative of X Coordinate vs Frame Number')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()

# Update with the correct path to your CSV file
data_analysis('provided_data.csv')
