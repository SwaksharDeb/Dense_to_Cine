import os
import shutil
import glob

# Paths to the Dense and Cine directories
dense_dir = 'Nifty Dense/TrainingData/'
cine_dir = 'Nifty Cine'

# Parent folder where subfolders will be created
parent_folder_train = 'Nifty/TrainingData'
parent_folder_val = 'Nifty/ValidationData'
# Get all .nii.gz files from both Dense and Cine directories
dense_files = glob.glob(os.path.join(dense_dir, '*_*.nii.gz'))
cine_files = glob.glob(os.path.join(cine_dir, '*_*.nii.gz'))

# Create the parent folder if it doesn't exist
os.makedirs(parent_folder_train, exist_ok=True)
os.makedirs(parent_folder_val, exist_ok=True)

# Combine all files into a single list
all_files = dense_files + cine_files

# Process each file
for nii_file in all_files:
    # Extract the base filename (without the full path)
    file_name = os.path.basename(nii_file)
    
    # Extract the subject_index from the filename (assuming it's the integer after the last underscore)
    subject_index = file_name.split('_')[-1].split('.')[0]
    
    # Create a subfolder for this subject_index inside the parent folder
    if subject_index == '98' or subject_index == '99':
        subject_dir = os.path.join(parent_folder_val, subject_index)
        os.makedirs(subject_dir, exist_ok=True)
    else:
        subject_dir = os.path.join(parent_folder_train, subject_index)
        os.makedirs(subject_dir, exist_ok=True)
    
    # Move the file into the corresponding subfolder under the parent folder
    shutil.copy2(nii_file, os.path.join(subject_dir, file_name))

print("Files organized successfully!")
