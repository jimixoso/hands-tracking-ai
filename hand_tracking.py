# Import files
import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up mediapipe Hands Module
mphands = mp.solutions.hands
hands = mphands.Hands(False)
mpDraw = mp.solutions.drawing_utils

# Initialize FPS variables
pTime = 0
cTime = 0

# Video Processing Loop
while True:

    # Read and Convert Frame
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand Detection
    results = hands.process(imgRGB)
    
    # Process Detected Hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


            # Draw Hand Landmarks
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    # Calculate FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display Processed Frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
