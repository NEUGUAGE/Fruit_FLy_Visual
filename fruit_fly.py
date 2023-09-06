import cv2
import numpy as np
import random
import time

# Create a VideoCapture object to access your camera (usually camera index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a BackgroundSubtractorMOG2 object for background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
chinese_char = "WT/FLY2/TRIAL1"

# Define the initial Chinese character size
chinese_char_size = 40

# Define the QR-like pattern
qr_pattern = [
    [1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 1]
]

# Initialize parameters for QR code and message animations
qr_size = 10
qr_alpha = 100
message_size = 100
message_x = 0
message_y = 0

# Number of frames
num_frames = 3

# Delay between each frame in milliseconds
frame_delay = 400  # You can adjust this delay

# Define the time (in seconds) after which the video will turn black and white
black_and_white_time = 5

# Keep track of the start time
start_time = time.time()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error")
        break

    # Check if it's time to turn the video to black and white
    if time.time() - start_time >= black_and_white_time:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the mask to create a binary image
    _, thresh = cv2.threshold(fg_mask, 56, 255, cv2.THRESH_BINARY)

    # Find contours of the moving object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mosaic mask of the same size as the frame
    mosaic_mask = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_NEAREST)
    mosaic_mask = cv2.resize(mosaic_mask, frame.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # Apply the mosaic effect to the entire frame
    mosaic_frame = cv2.bitwise_and(frame, mosaic_mask)

    # Draw colorful grid-like contours around the detected objects
    grid_size = 10
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for row in range(y, y + h, grid_size):
            for col in range(x, x + w, grid_size):
                if random.random() < 0.5:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.rectangle(mosaic_frame, (col, row), (col + grid_size, row + grid_size), color, -1)

    # Gradually increase the size and transparency of the QR-like pattern
    if qr_size < 200:
        qr_size += 2
        qr_alpha = min(qr_alpha + 5, 255)

    # Calculate the position to display the QR-like pattern
    qr_x = (frame.shape[1] - 7 * qr_size) // 2
    qr_y = (frame.shape[0] - 7 * qr_size) // 2

    # Overlay the QR-like pattern on the frame
    for i in range(len(qr_pattern)):
        for j in range(len(qr_pattern[i])):
            if qr_pattern[i][j] == 1:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(mosaic_frame, (qr_x + j * qr_size, qr_y + i * qr_size),
                              (qr_x + (j + 1) * qr_size, qr_y + (i + 1) * qr_size),
                              color, -1)

    # Calculate the position to display the message at the bottom
    message_x = (frame.shape[1] - len(chinese_char) * chinese_char_size) // 2
    message_y = frame.shape[0] - 20

    # Draw the message at the bottom of the frame
    cv2.putText(mosaic_frame, chinese_char, (message_x, message_y), cv2.FONT_HERSHEY_SIMPLEX,
                chinese_char_size / 24, (255, 255, 255), 2)

    # Display the final frame with all effects
    cv2.imshow('Video with Effects', mosaic_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


























