from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
# Replace 'yolov8n.pt' with the path to your custom-trained model if applicable
model = YOLO("yolov8n.pt")  # Example: "best.pt" for your trained model or "yolov8n.pt" for pretrained

# Function to detect objects in an image and print class labels
def detect_objects_and_print_classes(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Run object detection
    print("Running object detection...")
    results = model.predict(image, conf=0.2)  # Adjust confidence threshold if needed

    # Get class labels from the model
    class_names = model.names  # This holds the class labels (e.g., 'person', 'car', etc.)

    # Display detected objects with class labels
    detected = False
    for result in results:
        for box in result.boxes.data:
            detected = True
            class_id = int(box[-1])  # Class ID is the last value in the detection box
            confidence = box[4]  # Confidence score
            print(f"Detected Class: {class_names[class_id]} with Confidence: {confidence:.2f}")

    # If no objects are detected
    if not detected:
        print("No objects detected.")

    # Visualize results
    annotated_image = results[0].plot()  # Plot detections on the image
    cv2.imshow("YOLOv8 Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Specify the image path
    image_path = r"D:\Sajin\paper correction code\YoloV8 - Ashika\dataset\Deepfish\7490\train\7490_F3_f000019.jpg"

    # Check if the YOLO model has been loaded correctly
    if model:
        print("YOLO model loaded successfully.")
    else:
        print("Error: Failed to load YOLO model.")
        exit()

    # Call the detection function
    detect_objects_and_print_classes(image_path)
