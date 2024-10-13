import nilearn.plotting as plotting
from nilearn.image import load_img

# Load the NIfTI image
img = load_img('outputs/samples/BraTS2021_00495_t1_to_t2.nii.gz')

# Plot the image
plotting.plot_epi(img, title='T1 to T2 Diffusion Image')
plotting.show()


import nibabel as nib
import matplotlib.pyplot as plt

# Load the NIfTI file
img = nib.load('BraTS2021_00495_t1_to_t2.nii.gz')
data = img.get_fdata()

# Display the middle slice
slice_index = data.shape[2] // 2
plt.imshow(data[:, :, slice_index], cmap='gray')
plt.title('Middle Slice of T1 to T2 Diffusion Image')
plt.axis('off')
plt.show()