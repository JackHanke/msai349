import cv2
import mediapipe as mp
from utils.preprocessing import load_preprocessor, get_hand_edges, get_hand_bbox
from uuid import uuid4
from models.mlp import load_mlp_classifier_file
import os


def preprocess_frame(frame, img_size, preprocessor, custom_hands_obj):
    """
    Preprocess the frame to match the model's input.
    - Resize the frame to the training image size.
    - Flatten and normalize using the preprocessor.

    The frame needs to be in BGR format as input.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))  # Resize to match model input size
    frame = get_hand_edges(img=frame, custom_hands_obj=custom_hands_obj)
    reshaped_frame = frame.reshape(1, -1)  # Flatten the image
    return preprocessor.scaler.transform(reshaped_frame)


def main():
    # Configuration
    img_size = 64

    # Load trained model and preprocessor
    try:
        classifier = load_mlp_classifier_file()
        preprocessor = load_preprocessor()
    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        return

    # Initialize MediaPipe Hands for hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Start webcam video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Initialize the input_frame to None
        input_frame = None
        predicted_letter = "No Hand Detected"

        # Check for detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box for the hand
                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame_rgb.shape, padding_ratio=0.3)

                # Crop the hand region
                hand_roi = frame[y_min:y_max, x_min:x_max]

                # Visualize the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Use the hand ROI as the new input frame for prediction
                input_frame = hand_roi

                # Preprocess and predict
                try:
                    processed_frame = preprocess_frame(input_frame, img_size, preprocessor, custom_hands_obj=hands)
                    probs = classifier.predict(processed_frame, return_probs=True)
                    # if probs.max() < 0.75:
                    #     continue
                    predicted_letter = preprocessor.label_encoder.inverse_transform(probs.argmax(-1))[0]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_letter = "Unknown"

                # Display prediction on the frame
                cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 250), 2)
                break  # Only process the first detected hand

        # Show the video feed with predictions
        cv2.imshow("Sign Language Detection", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord(' '):  # Save image on 'space'
            if input_frame is not None:
                img_path = os.path.join("images", input("Specify image filename to save as: "))
                if "exit" in img_path:
                    continue
                cv2.imwrite(img_path, frame)
                print(f"Saved hand ROI to {img_path}.")
            else:
                print("No hand ROI to save.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()