import cv2
import pandas as pd
import argparse
import numpy as np

def calculate_movement(flow):
    """Calculate the total movement in the optical flow."""
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)

def filter_and_process_video(input_video_path, csv_path, output_video_path, movement_threshold):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    # Prepare to write the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Can't read first frame")
        return

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes (could be used to draw tracks)
    mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate movement
        movement = calculate_movement(flow)
        if movement > movement_threshold:
            out.write(frame)
        
        # Update previous frame
        prev_gray = gray

    cap.release()
    out.release()
    print(f"Video processed and saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter video to only include frames with high movement")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file with frame activity values (unused here)")
    parser.add_argument('--output_video', type=str, required=True, help="Path for output video")
    parser.add_argument('--movement_threshold', type=float, default=1000.0, help="Movement threshold to keep the frame")
    args = parser.parse_args()

    filter_and_process_video(args.input_video, args.csv_path, args.output_video, args.movement_threshold)

if __name__ == "__main__":
    main()
