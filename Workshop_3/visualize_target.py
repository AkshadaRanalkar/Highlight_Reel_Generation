import cv2
import pandas as pd
import argparse

def process_video(video_path, csv_path, output_path='output_video.mp4'):
    df = pd.read_csv(csv_path)
    frame_values = dict(zip(df['frame'], df['value']))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        value = frame_values.get(frame_number, 0)
        cv2.putText(frame, f"Frame {frame_number}: Value {value}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        frame_number += 1
    
    cap.release()
    out.release()
    print(f"Video processed and saved to {output_path}")

# For command line use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay values on video frames")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file with frame values")
    parser.add_argument('--output_path', type=str, default='output_video.mp4', help="Path for output video")
    args = parser.parse_args()
    
    process_video(args.video_path, args.csv_path, args.output_path)
