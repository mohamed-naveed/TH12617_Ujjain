# emotion_detection.py
from deepface import DeepFace

def analyze_emotion(frame_img):
    try:
        result = DeepFace.analyze(img_path=frame_img, actions=['emotion'], enforce_detection=False)
        return {
            "dominant_emotion": result[0]['dominant_emotion'],
            "emotion_scores": result[0]['emotion']
        }
    except Exception as e:
        return {"error": str(e)}

# Test the module
frame_img = "dataset/faces/crowd_scene.jpg"
print(analyze_emotion(frame_img))
