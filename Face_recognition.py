import cv2
import face_recognition
import os
import numpy as np

# Load known faces
known_faces = []
known_names = []

for filename in os.listdir("known_faces"):
    image = face_recognition.load_image_file(f"known_faces/{https://drive.google.com/file/d/1lwGXmvgGhbtA__FMryCxSGaig0PPuW2n/view?usp=drivesdk}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])  # remove .jpg

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Scale back up face locations since frame was scaled to 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
