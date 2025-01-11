from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your custom model if needed

# Function to detect objects in an image and print class labels
def detect_objects_and_print_classes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Run object detection
    results = model.predict(image, conf=0.5)

    # Get class labels from the model
    class_names = model.names

    # Display detected objects with class labels
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[-1])  # Class ID is the last value in the detection box
            print(f"Detected Class: {class_names[class_id]}")  # Print the class name

    # Visualize results
    annotated_image = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the image path
image_path = r"data\train\7117_Caranx_sexfasciatus_juvenile_f000000.jpg"
detect_objects_and_print_classes(image_path)
