import os

# Path to the folder containing your images
folder_path = './CAM_FRONT'

# Get a list of all files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# Sort the files by the timestamp part (if needed)
files.sort()

# Rename files sequentially to 0.jpg, 1.jpg, 2.jpg, etc.
for idx, file in enumerate(files):
    old_path = os.path.join(folder_path, file)
    new_name = f"{idx}.jpg"
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Files have been renamed successfully.")
