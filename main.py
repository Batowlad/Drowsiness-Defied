import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback to handle face landmarker results
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Open webcam
stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("Error: Could not open webcam")
    exit()

# Create the face landmarker and run detection inside the loop
with FaceLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0
    
    while True:
        ret, frame = stream.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Run async detection
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Increment timestamp (using frame count as simple timestamp)
        frame_timestamp_ms += 33  # ~30fps, roughly 33ms per frame
        
        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) == ord("q"):
            break

stream.release()
cv2.destroyAllWindows()
