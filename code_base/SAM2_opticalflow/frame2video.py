import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=30, resolution=(1920, 1080)):
    # Get all image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    
    # Check if there are images to process
    if not image_files:
        print("No images found!")
        return
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Read the image
        img = cv2.imread(image_path)
        
        # Resize image if necessary to match the resolution
        img_resized = cv2.resize(img, resolution)
        
        # Write the frame to the video
        out.write(img_resized)
    
    # Release the video writer
    out.release()
    print(f"Video saved as {output_video_path}")

# Example usage
image_folder = './results'  # Replace with your folder path
output_video_path = 'output_video.mp4'  # Output video path
create_video_from_images(image_folder, output_video_path, fps=30, resolution=(1920, 1080))
