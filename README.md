Project Title: GUARDIANAI: UNIFIED AI SAFETY & RECOVERY SYSTEM FOR MASS GATHERINGS
Team ID: TH12617

1. Overview

GuardianAI is an intelligent safety monitoring system designed to detect potential hazards and provide real-time alerts. It leverages computer vision and AI algorithms to analyze live camera feeds and notify users of suspicious activities. The system aims to enhance personal and public safety with minimal manual intervention.

2. Problem & Solution

Problem Statement:
Many public and private areas lack automated monitoring systems, leading to delayed response to accidents, theft, or unusual events. Existing solutions are either expensive or not adaptable to smaller environments.

Solution:
GuardianAI provides a cost-effective AI-powered monitoring system that can detect anomalies, alert authorities or users instantly, and maintain logs for future reference.

3. Logic & Workflow

 (1)Data Collection:
     Captures live video feed from cameras installed on-site.
     Collects image datasets for training AI models.

 (2)Processing:
     AI models analyze the video frames for unusual behavior or objects.
     Filters and preprocesses data to improve detection accuracy.

 (3)Output:
     Generates real-time alerts via notifications.
     Stores logs for each detected event with timestamp and location.

 (4)User Side:
     Receives alerts and event details on a web or mobile interface.
     Can view live camera feed and event history.

 (5)Admin Side:
     Manages camera configurations and user permissions.
     Monitors system performance and event analytics.

4. Tech Stack
GuardianAI/
│
├─ backend/       # All Python AI scripts + Flask API
│ ├── app.py      # Main Flask app
│ ├── face_recognition.py    # Face match logic
│ ├── object_detection.py    # YOLO-based item detection
│ ├── emotion_detection.py   # Emotion classification
│ ├── audio_analysis.py      # Panic detection from sound
├─ dataset/      # All training/testing sample files
│ ├── faces/     # Images of missing persons
│ ├── objects/   # Sample bags/phones
│ ├── audio/     # Crowd audio clips
├─ frontend/     # React.js
│ ├── dashboard_app/      # Web app code
├── models/               # Pretrained models (YOLOv5, DeepFace cache)
│
├── requirements.txt     # All Python libraries
└── README.md            # Project summary + setup guide
