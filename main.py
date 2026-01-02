import cv2

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    exit()

while (True):
    ret, frame = stream.read()
    if not ret:
        break
    
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
  