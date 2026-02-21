import csv
import os
import time
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

emotion_history = deque(maxlen=15)

# Load trained model
model = load_model("emotion_mobilenet_finetuned.h5")

emotion_labels = ["angry", "fear", "happy", "neutral", "sad"]

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

# 🔥 CSV Setup
csv_file = "emotion_output.csv"
file_exists = os.path.isfile(csv_file)

file = open(csv_file, mode="a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(["timestamp", "emotion", "confidence"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224,224))
        face = np.array(face, dtype=np.float32)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        emotion_history.append(prediction[0])

        avg_prediction = np.mean(emotion_history, axis=0)
        emotion_index = np.argmax(avg_prediction)
        confidence = float(np.max(avg_prediction))

        # 🔥 YAHAN CSV SAVE HO RAHA HAI
        timestamp = time.time()
        writer.writerow([
            timestamp,
            emotion_labels[emotion_index],
            confidence
        ])
        file.flush()

        label = f"{emotion_labels[emotion_index]} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()  # 🔥 Important
cv2.destroyAllWindows()
