import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# Define HSV color range for detecting a ping pong ball (adjust as needed)
lower_color = np.array([20, 100, 100])  
upper_color = np.array([40, 255, 255])  

prev_x, prev_y = None, None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2  # Get ball center
            
            # Calculate speed
            if prev_x is not None and prev_y is not None:
                curr_time = time.time()
                time_diff = curr_time - prev_time  # Time elapsed between frames
                
                if time_diff > 0:  
                    distance = ((cx - prev_x) ** 2 + (cy - prev_y) ** 2) ** 0.5  # Euclidean distance
                    speed = distance / time_diff  # Speed in pixels per second

                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                prev_time = curr_time

            prev_x, prev_y = cx, cy  # Update previous position
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Ball Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Ping Pong Ball Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
