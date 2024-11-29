import cv2
import mediapipe as mp
from utils.preprocessing import load_preprocessor
from models.sklearn import load_model
import warnings

warnings.filterwarnings('ignore')


def preprocess_frame(frame, img_size, preprocessor):
    """
    Preprocess the frame to match the model's input.
    - Resize the frame to the training image size.
    - Flatten and normalize using the preprocessor.
    """
    resized_frame = cv2.resize(frame, (img_size, img_size))  # Resize to match model input size
    reshaped_frame = resized_frame.reshape(1, -1)  # Flatten the image
    return preprocessor.scaler.transform(reshaped_frame)


def get_hand_bbox(hand_landmarks, frame_shape, padding_ratio):
    """
    Get a wider and taller bounding box for the detected hand.
    - Expands the bounding box by a given padding ratio.
    """
    h, w, _ = frame_shape
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    x_min, x_max = max(min(x_coords), 0), min(max(x_coords), w - 1)
    y_min, y_max = max(min(y_coords), 0), min(max(y_coords), h - 1)

    # Calculate padding
    width_padding = int((x_max - x_min) * padding_ratio)
    height_padding = int((y_max - y_min) * padding_ratio)

    # Expand bounding box with padding
    x_min = max(0, x_min - width_padding)
    y_min = max(0, y_min - height_padding)
    x_max = min(w, x_max + width_padding)
    y_max = min(h, y_max + height_padding)

    return x_min, y_min, x_max, y_max


def main():
    # Configuration
    img_size = 64  # Match your training image size
    model_name = "random_forest"

    # Load trained model and preprocessor
    classifier = load_model(model_name)
    preprocessor = load_preprocessor()

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

        # Check for detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box for the hand
                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape, padding_ratio=0.4)

                # Crop the hand region
                hand_roi = frame[y_min:y_max, x_min:x_max]

                # Visualize the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Use the hand ROI as the new input frame for prediction
                input_frame = hand_roi

                # Show the input image in a separate window
                # cv2.imshow("Input Image", input_frame)

                # Preprocess and predict
                try:
                    processed_frame = preprocess_frame(input_frame, img_size, preprocessor)
                    prediction = classifier.predict(processed_frame)
                    predicted_letter = preprocessor.label_encoder.inverse_transform(prediction)[0]
                except Exception as e:
                    print(f"Error: {e}")
                    predicted_letter = "Unknown"

                # Display prediction on the frame
                cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break  # Only process the first detected hand
        else:
            predicted_letter = "No Hand Detected"

        # Show the video feed with predictions
        cv2.imshow("Sign Language Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()