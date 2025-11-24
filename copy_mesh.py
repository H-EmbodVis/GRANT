import os
import shutil
import json

# Load the JSON file
with open('assets/datas/sample_seq.json', 'r') as file:
    data = json.load(file)

# Define the source base path and the destination base path
src_base_path = '/mnt/fillipo/scratch/masaccio/existing_datasets'
dest_base_path = './assets/datas'

# Function to copy files
def copy_file(file_path):
    if file_path.startswith(src_base_path):
        relative_path = os.path.relpath(file_path, src_base_path)
        dest_path = os.path.join(dest_base_path, relative_path)
        dest_dir = os.path.dirname(dest_path)
        
        # Create the destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy the file
        shutil.copyfile(file_path, dest_path)
        
        print(f"Copied {file_path} to {dest_path}")

# Iterate through the items in the JSON data
for key, value in data.items():
    if 'mesh_file' in value:
        copy_file(value['mesh_file'])
    
    if 'seg_file' in value:
        copy_file(value['seg_file'])

print("Files copied successfully.") 
