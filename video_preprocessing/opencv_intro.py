import cv2
import numpy as np
import argparse
import csv

def process_frame(frame, crop=None, resize=None):
    """
    Process a single video frame with optional cropping and resizing.

    Args:
        frame: Input video frame.
        crop: Crop parameters as percentages [x0, y0, width, height] (default: None).
        resize: Resize dimensions (width, height) (default: None).

    Returns:
        Processed video frame.
    """
    if crop:
        h, w = frame.shape[:2]
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(crop, [w, h, w, h])]
        frame = frame[y0:y0+crop_h, x0:x0+crop_w]

    if resize:
        frame = cv2.resize(frame, resize)

    return frame

def main():
    parser = argparse.ArgumentParser(description='Process video with optional CSV frame filtering')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--csv', type=str, help='Path to CSV file for frame filtering')
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'W', 'H'),
                        help='Crop video as percentage: x0 y0 width height')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Resize video to specific dimensions (width height)')
    args = parser.parse_args()

    # Open the input video
    input_video = cv2.VideoCapture(args.input_video)

    if not input_video.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(input_video.get(cv2.CAP_PROP_FPS))

    # Determine output resolution
    if args.resize:
        width, height = args.resize
    elif args.crop:
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(args.crop, [original_width, original_height, original_width, original_height])]
        width, height = crop_w, crop_h
    else:
        width, height = original_width, original_height

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, original_fps, (width, height))

    # Read CSV file if provided
    frame_filter = {}
    if args.csv:
        with open(args.csv, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                frame_filter[int(row['frame'])] = int(row['smoothed_value'])

        # Find the first non-zero value frame
        frame_number = next((frame for frame, value in frame_filter.items() if value != 0), 0)

        # Set the video capture to the first non-zero value frame
        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    else:
        frame_number = 0

    while True:
        ret, frame = input_video.read()

        if not ret:
            break

        # Check if we should process this frame
        if not frame_filter or frame_filter.get(frame_number, 0) == 1:
            # Process the frame
            processed_frame = process_frame(frame, args.crop, args.resize)

            # Write the processed frame to the output video
            output_video.write(processed_frame)

            # Display the processed frame
            cv2.imshow('Processed Video', processed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

