from ultralytics import YOLO
import cvzone
import cv2

# Load yolov8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = '123.mp4'
cap = cv2.VideoCapture(video_path)

# Read frames
# Initialize the ret variable
ret = True

# Run the loop
while ret:
    ret, frame = cap.read()
    if ret:
        # Initialize the class counts dictionary for each frame
        class_counts = {}

        # Run the model on the frame
        results = model.track(frame, persist=True)

        # Plot the results
        frame_ = results[0].plot()

        # Update the class counts
        for label, bbox in zip(results[0].boxes.cls, results[0].boxes.xyxy):
            class_name = model.names[label.item()]
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1

        # Display the class counts on the frame
        text = ""
        for class_name, count in class_counts.items():
            text += f"{class_name}: {count} "
        cv2.putText(frame_, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('frame', frame_)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):break

# Release the video capture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()