import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker
from norfair.drawing import draw_points

# Load the YOLOv8n model
model = YOLO('best.pt')  # Replace with your trained model path

# Function to convert YOLO detections to Norfair detections
def yolo_to_norfair(boxes, class_id=2, min_confidence=0.3):
    detections = []
    for box in boxes:
        if box.conf > min_confidence and int(box.cls) == class_id:  # Filter by class and confidence
            x_center = (box.xyxy[0] + box.xyxy[2]) / 2
            y_center = (box.xyxy[1] + box.xyxy[3]) / 2
            detections.append(
                Detection(points=[x_center, y_center], scores=[box.conf])
            )
    return detections

# Initialize Norfair tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Open video file
video_path = "gtavid1.mp4"  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Output video setup
output_path = "tracked_output.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Frame counter
frame_count = 0

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    frame_count += 1
    detections = []

    # Perform YOLO detection every 10th frame
    if frame_count % 10 == 0:
        results = model(frame)  # Run YOLO inference
        results = results[0]  # Access the first (and only) result
        detections = yolo_to_norfair(results.boxes)

    # Update tracker
    tracked_objects = tracker.update(detections=detections)

    # Draw tracked objects on the frame
    draw_points(frame, tracked_objects)

    # Write frame to output video
    out.write(frame)

    # Display frame (optional)
    # cv2.imshow("YOLO + Norfair Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracked video saved to {output_path}")
