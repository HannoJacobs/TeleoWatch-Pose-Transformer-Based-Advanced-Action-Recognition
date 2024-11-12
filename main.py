import argparse
from src.action_recognition import *

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Process a video file for action recognition."
)
parser.add_argument("--filename", help="Path to the video file")

# Parse the arguments
args = parser.parse_args()

FPS = 10.0
ar = ActionRecognition(
    fps=FPS,
    max_duration_sec=5.0,
    joints_to_use=[0, 5, 6, 7, 8, 11, 12, 13, 14],
    outlier_delta=100.0,
    percentage_joints_present=90,
    stillness_threshold=0.1,
    ar_yolo_model_file="models/yolov8l-pose.pt",
    yolo_conf=0.5,
    ar_model_conf=0.6,
    ar_model_file="models/fall_model.keras",
    dataset_labels={0: "fall", 1: "walk/stand", 999: "background"},
    show_frame=True,
    roi=[[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]],
)

if __name__ == "__main__":
    video = args.filename
    for i, frame in enumerate(video_frame_generator(video, fps=FPS)):
        id_and_action = ar.detect(frame)
