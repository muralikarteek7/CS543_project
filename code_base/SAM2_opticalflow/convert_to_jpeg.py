import os
from PIL import Image
import shutil

def convert_and_copy_images(input_dir, output_dir):
    """
    Converts PNG and JPG images in the input directory to JPEG format 
    and copies existing JPEG images to the output directory.
    
    Parameters:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where processed images will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpeg')
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                if filename.lower().endswith('.jpeg'):
                    # Copy JPEG files directly
                    shutil.copy(input_path, output_path)
                else:
                    # Convert PNG or JPG to JPEG
                    with Image.open(input_path) as img:
                        img = img.convert('RGB')  # Convert to RGB mode for JPEG compatibility
                        img.save(output_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = "/home/brije/sam2/frames"
output_directory = "/home/brije/sam2/frames_jpeg"
convert_and_copy_images(input_directory, output_directory)





