import cv2
import numpy as np

# --- Callback for trackbars ---
def nothing(x):
    pass

# --- Setup window and trackbars ---
cv2.namedWindow("Mask Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mask Tuner", 400, 300)

# Mask 1 trackbars
cv2.createTrackbar("L1H", "Mask Tuner", 0, 180, nothing)
cv2.createTrackbar("L1S", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("L1V", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("U1H", "Mask Tuner", 10, 180, nothing)
cv2.createTrackbar("U1S", "Mask Tuner", 255, 255, nothing)
cv2.createTrackbar("U1V", "Mask Tuner", 255, 255, nothing)

# Mask 2 trackbars
cv2.createTrackbar("L2H", "Mask Tuner", 170, 180, nothing)
cv2.createTrackbar("L2S", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("L2V", "Mask Tuner", 100, 255, nothing)
cv2.createTrackbar("U2H", "Mask Tuner", 180, 180, nothing)
cv2.createTrackbar("U2S", "Mask Tuner", 255, 255, nothing)
cv2.createTrackbar("U2V", "Mask Tuner", 255, 255, nothing)

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

    # Get values from trackbars for both masks
    l1_h = cv2.getTrackbarPos("L1H", "Mask Tuner")
    l1_s = cv2.getTrackbarPos("L1S", "Mask Tuner")
    l1_v = cv2.getTrackbarPos("L1V", "Mask Tuner")
    u1_h = cv2.getTrackbarPos("U1H", "Mask Tuner")
    u1_s = cv2.getTrackbarPos("U1S", "Mask Tuner")
    u1_v = cv2.getTrackbarPos("U1V", "Mask Tuner")

    l2_h = cv2.getTrackbarPos("L2H", "Mask Tuner")
    l2_s = cv2.getTrackbarPos("L2S", "Mask Tuner")
    l2_v = cv2.getTrackbarPos("L2V", "Mask Tuner")
    u2_h = cv2.getTrackbarPos("U2H", "Mask Tuner")
    u2_s = cv2.getTrackbarPos("U2S", "Mask Tuner")
    u2_v = cv2.getTrackbarPos("U2V", "Mask Tuner")

    lower1 = np.array([l1_h, l1_s, l1_v])
    upper1 = np.array([u1_h, u1_s, u1_v])
    lower2 = np.array([l2_h, l2_s, l2_v])
    upper2 = np.array([u2_h, u2_s, u2_v])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)
    result_combined = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Stack outputs
    row1 = np.hstack((frame, result1, result2))
    row2 = np.hstack((cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR),
                      result_combined))
    combined_view = np.vstack((row1, row2))

    cv2.imshow("Mask Tuner", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()