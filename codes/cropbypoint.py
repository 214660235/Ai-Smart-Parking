
# זה הקוד הטוב!!
import os
import uuid  # For generating unique filenames

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np


def cut_and_display(image_path, points, save_folder="car1233"):
    print(image_path)
    full_save_folder = os.path.join("path_to_your_desired_folder", save_folder)  # Change "path_to_your_desired_folder" to the desired path
    os.makedirs(full_save_folder, exist_ok=True)  # Ensure the directory exists

    # Remove existing images from the save folder
    existing_files = os.listdir(full_save_folder)
    for file in existing_files:
        os.remove(os.path.join(full_save_folder, file))

    # Cut and save the new images
    [cut_point(image_path, p, full_save_folder) for p in points]


def cut_point(image_path, p, save_folder):
    pil_image = Image.open(image_path)
    srgb_image = pil_image.convert("RGB")
    srgb_image = np.array(srgb_image)
    a, b, c, d = [int(p[i, 0]) for i in range(4)]
    a1, b1, c1, d1 = [int(p[i, 1]) for i in range(4)]
    srgb_image = srgb_image[min(a1, b1, c1, d1):max(a1, b1, c1, d1), min(a, b, c, d):max(a, b, c, d)]

    # Generate unique filename for the image
    unique_filename = str(uuid.uuid4()) + os.path.splitext(image_path)[-1]
    full_save_path = os.path.join(save_folder, unique_filename)

    # Save the new image
    plt.imsave(full_save_path, srgb_image)

    return srgb_image














