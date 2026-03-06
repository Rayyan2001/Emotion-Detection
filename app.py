import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

MODEL_PATH = r'E:\Emotional_Detection\ML_Model\best_model.h5'

try:
    model = load_model(MODEL_PATH)
    print("Emotion model loaded successfully")
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(1)


emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Haar Cascade not loaded")
    sys.exit(1)


def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))
    return face_img

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam not accessible")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam opened successfully")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        try:
            processed_face = preprocess_face(face_roi)
            predictions = model.predict(processed_face, verbose=0)
            emotion_index = int(np.argmax(predictions))
            emotion_text = emotion_labels.get(emotion_index, "Unknown")
            confidence = float(np.max(predictions) * 100)
        except Exception as e:
            continue

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        label = f"{emotion_text} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print("Program terminated")