import os
import glob
import nibabel as nib
import numpy as np

slice_num = 20
# Paths to the Dense and Cine directories
dense_dir = 'Nifty Dense/TrainingData/'
cine_dir = 'Nifty Cine'

# Parent folder where subfolders will be created
parent_folder_train = 'Nifty/TrainingData'
parent_folder_val = 'Nifty/ValidationData'

# Get all .nii.gz files from both Dense and Cine directories
dense_files = glob.glob(os.path.join(dense_dir, '*_*.nii.gz'))
cine_files = glob.glob(os.path.join(cine_dir, '*_*.nii.gz'))

# Create the parent folders if they don't exist
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
    file_name = file_name.split('.')[0]
    # Determine the appropriate parent folder
    if subject_index in ['98', '99']:
        parent_folder = parent_folder_val
    else:
        parent_folder = parent_folder_train
    
    # Load the NIfTI file
    img = nib.load(nii_file)
    data = img.get_fdata()
    
    # Get the original affine and header
    affine = img.affine
    header = img.header
    
    # Iterate through each slice
    for i in range(data.shape[2]):
        # Extract the 2D slice
        slice_data = data[:, :, i]
        
        # Reshape to (48, 48, 1)
        slice_data = slice_data.reshape(48, 48, 1)
        #zeros_data = np.zeros((48, 48, 19))
        #slice_data = np.concatenate([slice_data, zeros_data], axis=2)
        slice_data = np.repeat(slice_data, 20, axis=2)
        
        # Create a new NIfTI image for the 2D slice
        new_img = nib.Nifti1Image(slice_data, affine, header)
        
        # Save the 2D slice
        #new_file_name = f"{os.path.splitext(file_name)[0]}_slice{i:02d}.nii.gz"
        modality = file_name.split('_')[-2]
        new_file_name = f"{slice_num*int(subject_index)+i}_{modality}.nii.gz"

        # Create a subfolder for this subject_index inside the parent folder
        subject_dir = os.path.join(parent_folder, str(slice_num*int(subject_index)+i))
        os.makedirs(subject_dir, exist_ok=True)
        nib.save(new_img, os.path.join(subject_dir, new_file_name))

print("Files organized and converted to 2D slices successfully!")