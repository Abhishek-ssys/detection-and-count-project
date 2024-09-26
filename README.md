# YOLOv8 Object Detection and Tracking

This project demonstrates real-time object detection and tracking using the YOLOv8 model from `ultralytics` with OpenCV and `cvzone` for video processing.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project performs object detection on each frame of a video, using a pre-trained YOLOv8 model. It identifies and counts the detected objects, displaying both the bounding boxes and the object counts on each frame in real-time.

## Requirements

The project requires the following Python libraries:

- `ultralytics`: for YOLOv8 model (object detection and tracking)
- `opencv-python`: for video input, frame processing, and display
- `cvzone`: for utilities that enhance OpenCV operations

## Installation

First, clone this repository or download the script files, and ensure the required libraries are installed. 

1. Install the dependencies using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
