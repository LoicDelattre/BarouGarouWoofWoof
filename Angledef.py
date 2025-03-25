import cv2
import numpy as np

# --- Configuration ---
DEBUG_MODE = True  # Set to False to disable the debug view

# --- Helper Functions ---
def detect_red_objects(frame):
    """
    Detect red regions in the image.
    Returns the mask and the list of bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 97, 132])
    upper_red1 = np.array([6, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]

    return mask1, bounding_boxes

def calculate_horizontal_angle(x_center, frame_width, max_angle=45):
    """
    Calculate the angle from the center of the frame.
    Negative = left, Positive = right.
    """
    offset = x_center - (frame_width / 2)
    normalized_offset = offset / (frame_width / 2)
    angle = normalized_offset * max_angle
    return angle

# --- Main Loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    mask, boxes = detect_red_objects(frame)

    if DEBUG_MODE:
        debug_frame = frame.copy()
        for (x, y, w, h) in boxes:
            center_x = x + w // 2
            angle = calculate_horizontal_angle(center_x, frame_width)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{angle:.1f} deg", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"Detected object at angle: {angle:.1f} degrees")
        cv2.imshow("Red Object Detection", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()