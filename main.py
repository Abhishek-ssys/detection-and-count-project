from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = '11.mp4'
cap = cv2.VideoCapture(video_path)

# Read frames
ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        # Initialize the class counts
        human_count = 0
        object_count = 0

        # Run the model on the frame
        results = model.track(frame, persist=True)

        # Create a copy of the frame to draw bounding boxes
        frame_ = frame.copy()

        # Update the counts and draw bounding boxes manually
        for label, bbox in zip(results[0].boxes.cls, results[0].boxes.xyxy):
            class_name = model.names[label.item()]
            x1, y1, x2, y2 = map(int, bbox)  # Convert bounding box coordinates to integers

            if class_name.lower() == 'person':  # Assuming 'person' is the label for humans
                human_count += 1
            else:
                object_count += 1

            # Draw bounding box (without ID)
            color = (0, 255, 0) if class_name.lower() == 'person' else (255, 0, 0)
            cv2.rectangle(frame_, (x1, y1), (x2, y2), color, 2)

            # Add class name only (no ID)
            cv2.putText(frame_, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Add a white background rectangle for text display
        h, w, _ = frame_.shape
        cv2.rectangle(frame_, (0, h - 50), (w, h), (255, 255, 255), -1)

        # Display the counts of humans and objects
        text = f"Humans: {human_count} | Objects: {object_count}"
        cv2.putText(frame_, text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('frame', frame_)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
