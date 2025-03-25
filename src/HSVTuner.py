import cv2
import numpy as np

# --- Callback for trackbars ---
def nothing(x):
    pass

# --- Setup window and trackbars ---
cv2.namedWindow("Mask Tuner")
cv2.createTrackbar("Lower H", "Mask Tuner", 0, 180, nothing)
cv2.createTrackbar("Lower S", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("Lower V", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("Upper H", "Mask Tuner", 10, 180, nothing)
cv2.createTrackbar("Upper S", "Mask Tuner", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Mask Tuner", 255, 255, nothing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get values from trackbars
    l_h = cv2.getTrackbarPos("Lower H", "Mask Tuner")
    l_s = cv2.getTrackbarPos("Lower S", "Mask Tuner")
    l_v = cv2.getTrackbarPos("Lower V", "Mask Tuner")
    u_h = cv2.getTrackbarPos("Upper H", "Mask Tuner")
    u_s = cv2.getTrackbarPos("Upper S", "Mask Tuner")
    u_v = cv2.getTrackbarPos("Upper V", "Mask Tuner")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Stack original, mask, and result side-by-side
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, mask_bgr, result))

    cv2.imshow("Mask Tuner", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()