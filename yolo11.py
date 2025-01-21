
import cv2
from ultralytics import YOLO

# Set up the camera with USB
camera_id = 0  # Change this if your USB camera is on a different ID
video_capture = cv2.VideoCapture(camera_id)

# Set the resolution
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Load YOLOv8
#model = YOLO("yolo11n_ncnn_model")
#model = YOLO("best.pt")
model = YOLO("_v3.pt")
while True:
    # Capture a frame from the camera
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture frame")
        break
    
    # Run YOLO model on the captured frame and store the results
    results = model(frame, imgsz=640)
    
    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
