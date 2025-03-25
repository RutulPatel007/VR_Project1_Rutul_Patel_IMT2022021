import os
import re

directories = ["../../MSFD/1/face_crop", "../../MSFD/1/face_crop_segmentation"]

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  
            new_name = re.sub(r"_\d+", "", filename)  
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed in {directory}: {filename} -> {new_name}")


for dir_path in directories:
    rename_files(dir_path)
