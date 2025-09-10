#Project Title: GUARDIANAI: UNIFIED AI SAFETY & RECOVERY SYSTEM FOR MASS GATHERINGS
#Team ID: TH12617

##1. Overview

GuardianAI is an intelligent safety monitoring system designed to detect potential hazards and provide real-time alerts. It leverages computer vision and AI algorithms to analyze live camera feeds and notify users of suspicious activities. The system aims to enhance personal and public safety with minimal manual intervention.

##2. Problem & Solution

###Problem Statement:
Large public events like festivals, religious gatherings, and concerts face persistent safety challenges: people get lost, belongings go missing, and crowd panic can escalate into major incidents. Current surveillance systems often fail to provide timely, actionable insights for rapid emergency response or efficient lost and found operations.

###Solution:
It merges advanced computer vision and audio analytics to detect lost persons and belongings via real-time face and object recognition, and to identify early signs of panic or abnormal crowd behavior. Through emotion detection (facial analysis) and audio event detection (screams, chaos), Guardian AI is an AI-powered, all-in-one safety and recovery platform designed for mass gatherings. It empowers authorities with a live dashboard that combines map-based risk visualization, lost-and-found tracking, an instant alerting

##3. Logic & Workflow

###(1)Data Collection:
     Captures live video feed from cameras installed on-site.
     Collects image datasets for training AI models.

###(2)Processing:
     AI models analyze the video frames for unusual behavior or objects.
     Filters and preprocesses data to improve detection accuracy.

###(3)Output:
     Generates real-time alerts via notifications.
     Stores logs for each detected event with timestamp and location.

###(4)User Side:
     Receives alerts and event details on a web or mobile interface.
     Can view live camera feed and event history.

###(5)Admin Side:
     Manages camera configurations and user permissions.
     Monitors system performance and event analytics.
     
##4. Tech Stack

|    Component   |               Technology Used              |
|----------------|--------------------------------------------|
|   Frontend     |                React.js                    |
|    Backend     |      All Python AI scripts + Flask API     |
|   Database     |               MongoDB                      |
|  Requirements  |          All Python libraries              |
|     models     | Pretrained models (YOLOv5, DeepFace cache) |

##5. Future Scope
 
###(1)Advanced AI Models
       Integrate transformer-based face recognition (e.g., FaceNet + Vision Transformers) for higher accuracy in crowded environments. Use multi-modal AI combining image, audio, and movement patterns for better panic detection.
       
###(2)Scalable Cloud Deployment
       Deploy the entire system on cloud platforms like AWS / Azure / GCP. Use serverless APIs and auto-scaling to handle millions of pilgrims during Simhastha 2028.
       
###(3)Multilingual Voice Assistance
       Implement voice-based assistance in Hindi, English, Marathi, and other regional languages for guiding pilgrimssafely. Use speech-to-text for panic detection and text-to-speech for announcements.
       
###(4)Integration with Government & Police Systems
      Direct integration with local authorities for quick verification of missing persons and suspicious activities. Share real-time alerts with police, disaster response teams, and medical units.
      
###(5)Blockchain for Security
       Implement blockchain-based data storage to ensure secure handling of sensitive data like faces, belongings, and medicaldetails.

