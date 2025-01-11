import requests
import cv2
import matplotlib.pyplot as plt

# Replace these with your project details
API_KEY = "RqiLYoOdvYRVo9H2MTOc"
PROJECT_NAME = "fish"
VERSION = "v3"

# Image file to process
IMAGE_PATH = r"D:\Sajin\paper correction code\YoloV8 - Ashika\dataset\data\fish.v3i.yolov8\train\images"

# Roboflow API endpoint
INFERENCE_URL = f"https://detect.roboflow.com/fisg/v2?api_key=RqiLYoOdvYRVo9H2MTOc"

def get_predictions(image_path):
    """
    Send an image to the Roboflow API for YOLOv inference.
    """
    with open(image_path, "rb") as image_file:
        response = requests.post(INFERENCE_URL, files={"file": image_file})
    if response.status_code == 200:
        return response.json()  # Return the JSON response with predictions
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def draw_predictions(image_path, predictions):
    """
    Draw bounding boxes and class labels from predictions onto the image.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parse predictions
    for pred in predictions["predictions"]:
        x, y, width, height = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]

        # Calculate bounding box coordinates
        x1, y1 = x - width // 2, y - height // 2
        x2, y2 = x + width // 2, y + height // 2

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Main Workflow
try:
    # Step 1: Get predictions from the Roboflow API
    predictions = get_predictions(IMAGE_PATH)

    # Step 2: Visualize predictions on the input image
    draw_predictions(IMAGE_PATH, predictions)
except Exception as e:
    print(e)
