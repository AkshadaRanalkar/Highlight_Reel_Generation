import cv2
import numpy as np
import argparse

def create_transition(cap, start_frame, end_frame, transition_type='fade', duration_frames=30):
    """
    Create a transition between two frames in a video.

    Parameters:
    cap: cv2.VideoCapture object
    start_frame: int, starting frame number
    end_frame: int, ending frame number
    transition_type: str, type of transition ('fade', 'wipe_left', 'wipe_right', 'dissolve')
    duration_frames: int, number of frames for the transition

    Returns:
    list of frames containing the transition
    """
    # Save original position
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Get the two frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame2 = cap.read()
    
    if not ret or frame1 is None or frame2 is None:
        raise ValueError("Could not read frames")

    # Convert frames to float32 for better transition quality
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)

    transition_frames = []

    for i in range(duration_frames):
        progress = i / (duration_frames - 1)
        
        if transition_type == 'fade':
            frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        elif transition_type == 'wipe_left':
            width = frame1.shape[1]
            cut_point = int(width * progress)
            frame = frame1.copy()
            frame[:, :cut_point] = frame2[:, :cut_point]
        elif transition_type == 'wipe_right':
            width = frame1.shape[1]
            cut_point = int(width * (1 - progress))
            frame = frame1.copy()
            frame[:, cut_point:] = frame2[:, cut_point:]
        elif transition_type == 'dissolve':
            mask = np.random.random(frame1.shape[:2]) < progress
            mask = np.stack([mask] * 3, axis=2)
            frame = np.where(mask, frame2, frame1)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
        
        frame_uint8 = frame.astype(np.uint8)
        transition_frames.append(frame_uint8)

    # Restore original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    return transition_frames

def save_transitions(video_path, output_path, start_frame, end_frame, transition_type, duration_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Generate transition frames
    transition_frames = create_transition(cap, start_frame, end_frame, transition_type, duration_frames)

    # Write transition frames to output
    for frame in transition_frames:
        out.write(frame)

    # Optional: Add padding to extend video duration
    for _ in range(fps * 2):  # Add 2 seconds of padding
        out.write(transition_frames[-1])  # Repeat the last frame

    cap.release()
    out.release()
    print(f"Transition video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video transitions")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--output_path', type=str, required=True, help="Path for the output video")
    parser.add_argument('--start_frame', type=int, required=True, help="Start frame for transition")
    parser.add_argument('--end_frame', type=int, required=True, help="End frame for transition")
    parser.add_argument('--transition_type', type=str, default='fade', help="Type of transition (fade, wipe_left, wipe_right, dissolve)")
    parser.add_argument('--duration_frames', type=int, default=60, help="Number of frames for the transition")

    args = parser.parse_args()

    save_transitions(args.video_path, args.output_path, args.start_frame, args.end_frame, args.transition_type, args.duration_frames)
