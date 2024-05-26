import cv2
import os
import shutil
from skimage.metrics import structural_similarity as compare_ssim


def classify_and_move_images(
    model_image_path, folder_path, destination_folder, similarity_threshold=40
):
    # Load the model image
    model_image = cv2.imread(model_image_path)

    # Verify if the model image is loaded successfully
    if model_image is None:
        raise ValueError(f"Model image at path {model_image_path} could not be loaded.")

    # Convert the model image to grayscale
    gray_model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
    model_height, model_width = gray_model_image.shape

    results = {}

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load the current image
        image = cv2.imread(file_path)

        # Skip if the file is not an image
        if image is None:
            continue

        # Convert the current image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the current image to match the model image dimensions
        resized_image = cv2.resize(gray_image, (model_width, model_height))

        # Calculate the structural similarity index (SSIM) between the images
        (score, diff) = compare_ssim(gray_model_image, resized_image, full=True)
        similarity_percentage = score * 100

        # Classify the images based on the similarity percentage
        if similarity_percentage >= similarity_threshold:
            classification = "Similar"
            # Move the image to the destination folder
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(file_path, destination_path)
        else:
            classification = "Different"

        results[filename] = (classification, similarity_percentage)

    return results


# Example usage
model_image_path = "img21.jpeg"
folder_path = "prints"
destination_folder = "similar_images"

try:
    results = classify_and_move_images(
        model_image_path, folder_path, destination_folder, similarity_threshold=40
    )
    for filename, (classification, similarity_percentage) in results.items():
        print(
            f"Image {filename} is {classification} with a similarity percentage of {similarity_percentage:.2f}%."
        )
except ValueError as e:
    print(e)
