import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data (adjust path if needed)
df = pd.read_csv('provided_data.csv')

# Calculate the min and max values for the assumed X and Y coordinate columns (columns 1 and 2)
x_min, x_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()
y_min, y_max = df.iloc[:, 2].min(), df.iloc[:, 2].max()

print(f"x_min = {x_min} , x_max = {x_max} , y_min = {y_min} , y_max = {y_max}")

# Feature 1: Distance Traveled
# Calculate the distance between consecutive points and then get the cumulative sum
df['distance'] = ((df.iloc[:, 1].diff()**2 + df.iloc[:, 2].diff()**2)**0.5).cumsum()

# Print the first few rows to verify the feature calculation
print("\nDistance Traveled Feature:")
print(df[['distance']].head())

# Feature 2: Angle Change
# Calculate displacement vectors between consecutive points
df['displacement_x'] = df.iloc[:, 1].diff()
df['displacement_y'] = df.iloc[:, 2].diff()

# Calculate the angle (direction) of the displacement vector using arctangent
df['angle'] = np.arctan2(df['displacement_y'], df['displacement_x'])

# Calculate the change in angle between consecutive points
df['angle_change'] = df['angle'].diff().abs()

# Print the first few rows to verify the angle change calculation
print("\nAngle Change Feature:")
print(df[['angle_change']].head())

# Visualize the angle changes
plt.figure(figsize=(8, 6))
plt.plot(df.index, df['angle_change'], label='Angle Change', color='green')
plt.xlabel('Frame Number')
plt.ylabel('Angle Change')
plt.title('Angle Change Over Time')
plt.legend()
plt.savefig('Angle_Change_Visualization.png')  # Save the plot
plt.show()

# Visualize the distance traveled over time
plt.figure(figsize=(8, 6))
plt.plot(df.index, df['distance'], label='Cumulative Distance Traveled', color='blue')
plt.xlabel('Frame Number')
plt.ylabel('Cumulative Distance')
plt.title('Distance Traveled Over Time')
plt.legend()
plt.savefig('Distance_Traveled.png')
plt.show()
