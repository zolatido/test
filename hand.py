import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_hands = mp.solutions.hands
try:
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
except:
    hands = None

mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks_data = []
    combined_landmarks = []
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmark_data = [landmark.x, landmark.y]
                if hasattr(landmark, 'z'):
                    landmark_data.append(landmark.z)
                landmarks.extend(landmark_data)

            # Pad with zeros if needed to ensure a total length of 126
            landmarks += [0.0] * (126 - len(landmarks))

            landmarks_data.append(landmarks)
            combined_landmarks.extend(landmarks)

    return combined_landmarks, results  # Return 'results' along with 'combined_landmarks'

def predict_hand_action(frame):
    landmarks, results = extract_landmarks(frame)
    if not landmarks:
        return None

    predictions = []
    for hand_landmarks in results.multi_hand_landmarks:
        landmarks_arr = np.array([landmarks])
        reshaped_landmarks = landmarks_arr.reshape((1, 1, landmarks_arr.shape[1]))

        prediction = model.predict(reshaped_landmarks)

        # Check if the maximum value in the prediction array is greater than 0.5
        if np.max(prediction) > 0.5:
            predictions.append("Correct Position")
        else:
            predictions.append("Incorrect Position")

    return predictions



# Load your machine learning model
model_path = r"D:\Downloads\lstm_1-new.h5"
model = tf.keras.models.load_model(model_path)

# Open the camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict and display result
    if hands is not None:
        landmarks, results = extract_landmarks(frame)
        if landmarks:
            prediction = predict_hand_action(frame)
            if prediction:
                print(prediction)

            # Display the frame with landmarks and predictions
            frame_with_landmarks = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks, prediction in zip(results.multi_hand_landmarks, predict_hand_action(frame)):
                    mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame_with_landmarks, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with landmarks and predictions
            cv2.imshow('Hand Landmarks', frame_with_landmarks)

    else:
        # Display the original frame if hands module is not available
        cv2.imshow('Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if hands is not None:
    hands.close()
