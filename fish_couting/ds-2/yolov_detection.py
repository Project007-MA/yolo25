from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model (replace with your custom-trained model path if applicable)
model = YOLO('yolov8n.pt')  # You can replace this with yolov8s.pt, yolov8m.pt, or your custom model.

# Function to detect objects in an image
def detect_objects_in_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Run object detection
    results = model.predict(image, conf=0.5)  # Set confidence threshold (0.5 is default)
    
    # Display detected objects
    for result in results:
        print(f"Detected objects in {image_path}: {result.boxes.data}")
    
    # Visualize results on the image
    annotated_image = results[0].plot()  # Draw detections on the original image
    
    # Show the annotated image
    cv2.imshow("YOLOv8 Detection", annotated_image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

# Main function to process all images in a directory
def process_images_in_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        return

    # Loop through all files in the directory
    for file_name in os.listdir(directory_path):
        image_path = os.path.join(directory_path, file_name)
        # Process only image files
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {image_path}")
            detect_objects_in_image(image_path)
        else:
            print(f"Skipping non-image file: {file_name}")

# Specify the directory containing your test images
directory_path = r"D:\Sajin\paper correction code\YoloV8 - Ashika\dataset\data\test"

# Process all images in the directory
process_images_in_directory(directory_path)
