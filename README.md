
# Highlight Reel Video 

---

## Output Highlights

Experience volleyball match highlights featuring advanced prediction smoothing, transitions, and insightful visualizations:  
[![Watch Highlights on YouTube](https://youtu.be/YOlFYVxnS3Q?si=r3UQ4Zk4bFgEERsU)

---

## Overview
The **An Video Highlight Reel Generating System**  utilizes a combination of trajectory analysis, machine learning classification, and video processing to generate engaging highlights from match footage. 
This system streamlines the workflow for analysts, coaches, and content creators by automating event detection and video production.

---

## Table of Contents
1. [Project Workflow](#project-phases)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Model and Processing Details](#model-details)
7. [Results](#results)
8. [Video Transition Details](#video-transition)
9. [Evaluation and Metrics](#evaluation-and-metrics)
10. [Acknowledgments](#acknowledgments)

---

## Project Workflow

The project is divided into several stages:

1. **Data Analysis and Feature Engineering:** Perform data cleaning and derive additional features using scripts like data_analysis.py.
Animate ball trajectories using animation.py.
2. **Model Training and Prediction:** Train a classification model using Time_Classification1.py.
Visualize model predictions with visualize_target.py.
3. **Prediction Smoothing:** Smooth raw predictions using techniques like moving averages, Gaussian filters, and expansion (handled in filter_predictions.py).
4. **Highlight Generation:** Generate highlights from smoothed predictions, with seamless transitions applied to key events (video_transition.py).
5. **Final Output and Evaluation:** Save results and visualize comparisons between raw, smoothed predictions, and ground truth data.

---

## Features
- **Feature-Rich Analysis:** Extract meaningful features such as ball position, trajectory, and interaction with boundaries.
- **Advanced Smoothing:**: Techniques like median, Gaussian, and moving average filters ensure stable event detection.
- **Highlight Transitions:** High-quality transitions, including fade, slide, and dissolve effects, for professional output.
- **Customizable Parameters:** User-configurable settings for model tuning, smoothing window size, and transition effects.

---

## Installation

### Requirements

Before running the project, ensure that you have the following dependencies installed:

- Python 3.7+
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

### Steps to Install

1. Clone the repository:

2. Install dependencies:

3. Ensure video and data files are located in the data/ folder.
   
---

## Usage

### 1. Preprocess and Visualize Data

- **Clean and analyze match data:**
 

- **Animate ball trajectories in the match video:**


- **Feature Visualization**: Visualize the custom features with ball tracking in a video using `animation.py`:

### 2. Train Classification Model

- **Train the classifier to predict key match events:** 

- **Visualize predictions:** 

### 3. Smooth Predictions:
   - Apply smoothing techniques to reduce noise in predictions:

### 4. Highlight Generation:
   - Once the model is trained, run the opencv_intro script to create the highlights from a video:


### 5. Video Transition:
   - Produce highlights with transitions:

---

## File Structure


Video Highlight Reel Generating System/
│
├── data/                         # Contains raw data and videos
│   ├── video.mp4                 # Input video file
│   ├── provided_data.csv         # Raw trajectory data
│   └── target.csv                # Classification target data
│
├── scripts/                      # Python scripts for different stages
│   ├── data_preprocessing/       # Data collection and preprocessing
│   │   ├── data_analysis.py   # Initial data cleaning and visualization
│   │   └── animation.py          # Visualize custom features with ball tracking
│   │
│   ├── model_training/           # Model selection and training
│   │   ├── Time_Classification1.py    # Train XGboost classifier
│   │   └── visualize_target.py     # Visualize model predictions on video
│   │
│   ├── video_processing/         # Video smoothing and transitions
│   │   ├── filter_predictions.py  # Apply advanced filtering techniques
│   │   ├── opencv_intro.py        # Highlight generation with OpenCV
│   │   └── video_transition.py    # Apply transitions between video segments
│
├── results/                      # Output files
│   ├── smoothed_predictions.csv  # Smoothed model predictions
│   ├── output_video.mp4            # Video without transitions
│   └── highlight_reel_transitions.mp4        # Video with transitions
│
└── requirements.txt              # Project dependencies


---

## Model Processing Details

Model Overview:

1. **Classifier:** 
   - XGBoost (eXtreme Gradient Boosting)
     
2. **Feature Engineering:** 
   - Incorporates ball trajectory features, aspect ratio, size, and additional engineered features like time-lagged values.

3. **Hyperparameter Optimization:** 
   - Fine-tuned learning rate, maximum depth, and the number of estimators for optimal classification accuracy.

---

## Smoothing

- **Techniques:** Applies advanced techniques such as Median filtering, Gaussian smoothing, and moving average to stabilize predictions.
- **Implementation:** Configured through filter_predictions.py using the PredictionProcessor class, which applies the best-suited smoothing techniques.

---

## Transitions

- **Effect Types:** Includes effects like fade, slide, and dissolve to create visually appealing transitions between key moments.
- **Implementation:** Handled in video_transition.py using OpenCV's advanced blending techniques.

---


## Results

1. **Model Performance**:
   - The XGboost classifier was trained and validated with various feature sets, yielding impressive classification accuracy. Hyperparameter optimization improved the **weighted F1 score**, making the model more reliable for detecting key events in the game. below is the result from the classification report

     <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px; margin-bottom: 20px">
        <img src="results/workshop_4_output.png" alt="Classification Report" width="400" height="250"/>
        <img src="results/f1_heatmap.png" alt="F1 Score HeatMap" width="400" height="250"/>
     </div>

2. **Prediction Smoothing:**
   - The system generates smoothed predictions by comparing raw predictions with the ground truth targets. The results are visualized.
     
    <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px; margin-bottom: 20px">
        <img src="results/predictions_comparison.png" alt="Comparison of Raw and smooth Prediction" width="700" height="300"/>
    </div>
   
3. **Highlight Video:**
   - Highlights with smooth transitions are saved. Transition effects like fade, slide, and dissolve enhance the presentation quality of the video.

    <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px; margin-bottom: 20px">
        <img src="results/filter_predictions_output.png" alt="Output of filter Prediction" width="700" height="300"/>
    </div>

4. **Video Transition Details**
   - The video transitions were visually appealing, adding a professional touch to the highlight reel.
   - Transitions: Transitions between highlights are generated using OpenCV.
   - Fade: Gradual blending of two frames to create a smooth effect.
   - Slide: Left-to-right or right-to-left frame transitions for a dynamic effect.
   - Dissolve: Pixel-level blending based on randomized masking to create a seamless transition.
   - [Watch the Full Video Highlight on YouTube](https://youtu.be/YOlFYVxnS3Q?si=IVNFbEKPjGHvNNZs)

---

## Evaluation and Metrics

The model was evaluated using the following metrics:

- **Precision**: Measures the accuracy of the positive predictions.
- **Recall**: Measures the ability of the model to find all relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measures the trade-off between true positive rate and false positive rate.

---

## Acknowledgments

- OpenCV: Used for video processing and transition effects.
- XGBoost: For robust and accurate event classification.
- Matplotlib: For visualizing predictions and smoothed results.
- Pandas: For data manipulation and handling.
