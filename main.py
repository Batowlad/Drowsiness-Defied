import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2 
import numpy as np

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Store the latest detection result (updated by async callback)
latest_result = None

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw face landmarks on the image."""
    if detection_result is None or not detection_result.face_landmarks:
        return rgb_image
    
    annotated_image = np.copy(rgb_image)
    
    for face_landmarks in detection_result.face_landmarks:
        # Convert to proto format for drawing
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])
        
        # Draw face mesh tesselation
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        
        # Draw face mesh contours
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())
        
        # Draw irises
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    
    return annotated_image

# Callback to handle face landmarker results
def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    
    if result.face_landmarks:
        print(f"\n=== Frame {timestamp_ms}ms ===")
        print(f"Detected {len(result.face_landmarks)} face(s)")
        
        # Print landmarks (showing first 5 of 478)
        for i, face in enumerate(result.face_landmarks):
            print(f"\nFace {i}: {len(face)} landmarks")
            for j, lm in enumerate(face[:5]):  # First 5 landmarks
                print(f"  Landmark {j}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")
            print("  ...")
        
        # Print blendshapes (requires output_face_blendshapes=True)
        if result.face_blendshapes:
            for i, blendshapes in enumerate(result.face_blendshapes):
                print(f"\nFace {i} Blendshapes ({len(blendshapes)} categories):")
                for bs in blendshapes:
                    print(f"  {bs.category_name}: {bs.score:.4f}")
        
        # Print transformation matrices (requires output_facial_transformation_matrixes=True)
        if result.facial_transformation_matrixes:
            for i, matrix in enumerate(result.facial_transformation_matrixes):
                print(f"\nFace {i} Transformation Matrix:")
                print(matrix)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_face_blendshapes=True,
    result_callback=on_result)

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
        
        # Increment timestamp
        frame_timestamp_ms += 33  # ~30fps
        
        # Draw landmarks on the frame
        annotated_frame = draw_landmarks_on_image(frame_rgb, latest_result)
        
        # Convert back to BGR for OpenCV display
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Face Landmarks", display_frame)

        if cv2.waitKey(1) == ord("q"):
            break

stream.release()
cv2.destroyAllWindows()
