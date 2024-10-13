import os
import numpy as np
import nibabel as nib
import cv2

def extract_frame_from_nifti(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Extract the 0th frame of the last index (assuming shape is 48,48,20)
    frame = data[:,:,0]
    
    # Normalize to 0-255 range for video
    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
    
    return frame

def create_video_from_years(base_folder, start_year, end_year, output_file, fps=10):
    frames = []
    for year in range(start_year, end_year + 1):
        folder = os.path.join(base_folder, str(year))
        nifti_file = os.path.join(folder, str(year)+'_cine.nii.gz')
        
        if os.path.exists(nifti_file):
            frame = extract_frame_from_nifti(nifti_file)
            frames.append(frame)
            print(f"Processed: {nifti_file}")
        else:
            print(f"File not found: {nifti_file}")
    
    if not frames:
        print(f"No frames extracted for years {start_year}-{end_year}")
        return
    
    # Create video
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert grayscale to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        out.write(rgb_frame)
    
    out.release()
    print(f"Video created: {output_file}")

def process_year_ranges(base_folder):
    create_video_from_years(base_folder, 1960, 1979, os.path.join('Target Cine Videos', 'video_1960_1979.mp4'))
    create_video_from_years('Target Cine Videos', 1980, 1999, os.path.join('Target Cine Videos', 'video_1980_1999.mp4'))

# Usage
base_folder = 'Nifty/ValidationData'
process_year_ranges(base_folder)