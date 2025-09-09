# face_recognition.py
from deepface import DeepFace

def verify_face(input_img, frame_img):
    try:
        result = DeepFace.verify(img1_path=input_img, img2_path=frame_img, enforce_detection=False)
        return {
            "match": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"]
        }
    except Exception as e:
        return {"error": str(e)}

# Test the module
input_img = "dataset/faces/missing_person.jpg"
frame_img = "dataset/faces/cctv_frame.jpg"

print(verify_face(input_img, frame_img))
