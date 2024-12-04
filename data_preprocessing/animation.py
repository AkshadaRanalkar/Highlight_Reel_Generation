import pandas as pd
import cv2
import numpy as np

def create_animation(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['frame', 'x_coord', 'y_coord', 'feature1', 'feature2', 'optional_feature']

    video_width, video_height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Changed to XVID for better compatibility
    out = cv2.VideoWriter('animation.avi', fourcc, 30.0, (video_width, video_height))  # Changed to .avi

    x_min, x_max = df['x_coord'].min(), df['x_coord'].max()
    y_min, y_max = df['y_coord'].min(), df['y_coord'].max()

    for _, row in df.iterrows():
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        x = int((row['x_coord'] - x_min) / (x_max - x_min) * (video_width - 20) + 10)
        y = int((row['y_coord'] - y_min) / (y_max - y_min) * (video_height - 20) + 10)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Frame: {int(row['frame'])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print("Writing frame shape:", frame.shape)  # Diagnostic print

        out.write(frame)  # Attempt to write frame

    out.release()
    print("Animation saved as 'animation.avi'")

create_animation('provided_data.csv')
