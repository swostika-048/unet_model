import os
import cv2

def grayscale_to_binary(image_path, output_folder, threshold=0.2):
    # Read the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary_image = cv2.threshold(grayscale_image, threshold * 255, 255, cv2.THRESH_BINARY)

    # Extract the filename and extension
    filename = os.path.basename(image_path)
    filename_no_extension = os.path.splitext(filename)[0]
    output_filename = filename_no_extension + "_binary.jpg"

    # Save the binary image to the output folder
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, binary_image)

def convert_images_in_folder(input_folder, output_folder, threshold=0.5):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print("folder exist")

    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tif")]

    # Process each image in the input folder
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        grayscale_to_binary(image_path, output_folder, threshold)

# Example usage:
input_folder = "driveway_dataset/output_2000_mask"
output_folder = "driveway_dataset/binary_output_2000_mask"
convert_images_in_folder(input_folder, output_folder, threshold=0.5)
