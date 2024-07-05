import numpy as np
import cv2
import pickle

# Load your trained model
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

# Define image dimensions expected by your model
image_dim = (32, 32)  # Use the correct dimensions that your model expects

# Function to preprocess an image
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couqld not open webcam.")
    exit()

while True:
    success, img_for_prediction = cap.read()

    # Check if the frame was captured successfully
    if not success:
        print("Error: Failed to capture image from webcam.")
        break

    # Resize and preprocess the captured frame
    img_for_prediction = cv2.resize(img_for_prediction, image_dim)
    img_for_prediction = preProcessing(img_for_prediction)
    img_for_prediction = np.expand_dims(img_for_prediction, axis=-1)  # Add channel dimension
    img_for_prediction = np.expand_dims(img_for_prediction, axis=0)  # Add batch dimension

    # Perform prediction
    predictions = model.predict(img_for_prediction)

    # Example: Print the predicted class (assuming it's a classification model)
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")

    # Display the frame
    cv2.imshow("Webcam", img_for_prediction[0])

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
