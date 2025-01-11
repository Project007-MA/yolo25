import os
list1=[]
for i in os.listdir(r"C:\Users\ADMIN\Music\detect_fish images"):
    list1.append(i)

import os
import shutil

# List of target image names
target_images = list1


source_directory = r"C:\Users\ADMIN\Downloads\dataset\Fish1.v1i.yolov9\train\images"
destination_directory = r"C:\Users\ADMIN\Music\selected_fish"

os.makedirs(destination_directory, exist_ok=True)

for image_name in target_images:
    source_path = os.path.join(source_directory, image_name)
    destination_path = os.path.join(destination_directory, image_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {image_name} to {destination_directory}")
    else:
        print(f"{image_name} does not exist in the source directory")
