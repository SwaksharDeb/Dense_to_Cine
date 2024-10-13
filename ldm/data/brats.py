import os
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset

# brats_transforms = transforms.Compose(
#     [
#         transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
#         transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"], allow_missing_keys=True),
#         transforms.Lambdad(keys=["t1", "t1ce", "t2", "flair"], func=lambda x: x[0, :, :, :]),
#         #transforms.AddChanneld(keys=["t1", "t1ce", "t2", "flair"]),
#         transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
#         transforms.EnsureTyped(keys=["t1", "t1ce", "t2", "flair"]),
#         transforms.Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAI", allow_missing_keys=True),
#         transforms.CropForegroundd(keys=["t1", "t1ce", "t2", "flair"], source_key="t1", allow_missing_keys=True),
#         transforms.SpatialPadd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=(160, 160, 128), allow_missing_keys=True),
#         transforms.RandSpatialCropd( keys=["t1", "t1ce", "t2", "flair"],
#             roi_size=(160, 160, 128),
#             random_center=True, 
#             random_size=False,
#         ),
#         transforms.ScaleIntensityRangePercentilesd(keys=["t1", "t1ce", "t2", "flair"], lower=0, upper=99.75, b_min=0, b_max=1),
#     ]
# )

# brats_transforms = transforms.Compose(
#     [
#         transforms.LoadImaged(keys=["cine", "dense"], allow_missing_keys=True),
#         transforms.EnsureChannelFirstd(keys=["cine", "dense"], allow_missing_keys=True),
#         transforms.Lambdad(keys=["cine", "dense"], func=lambda x: x[0, :, :, :]),
#         transforms.EnsureChannelFirstd(keys=["cine", "dense"]),
#         transforms.EnsureTyped(keys=["cine", "dense"]),
#         transforms.Orientationd(keys=["cine", "dense"], axcodes="RAI", allow_missing_keys=True),
#         transforms.CropForegroundd(keys=["cine", "dense"], source_key="dense", allow_missing_keys=True),
#         transforms.SpatialPadd(keys=["cine", "dense"], spatial_size=(48, 48, 20), allow_missing_keys=True),
#         transforms.RandSpatialCropd( keys=["cine", "dense"],
#             roi_size=(160, 160, 128),
#             random_center=True, 
#             random_size=False,
#         ),
#         transforms.ScaleIntensityRangePercentilesd(keys=["cine", "dense"], lower=0, upper=99.75, b_min=0, b_max=1),
#     ]
# )

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["cine", "dense"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["cine", "dense"], allow_missing_keys=True),
        transforms.Lambdad(keys=["cine", "dense"], func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["cine", "dense"]),
        transforms.EnsureTyped(keys=["cine", "dense"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["cine", "dense"], lower=0, upper=100, b_min=0, b_max=1),
    ]
)

# def get_brats_dataset(data_path):
#     transform = brats_transforms
    
#     data = []
#     for subject in os.listdir(data_path):
#         sub_path = os.path.join(data_path, subject)
#         if os.path.exists(sub_path) == False: continue
#         t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
#         t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz") 
#         t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 
#         flair = os.path.join(sub_path, f"{subject}_flair.nii.gz") 
#         seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

#         data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject})
                    
#     print("num of subject:", len(data))

#     return MonaiDataset(data=data, transform=transform)

def get_brats_dataset(data_path):
    transform = brats_transforms
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        dense = os.path.join(sub_path, f"cardiac_mri_dense_{subject}.nii.gz") 
        #dense = os.path.join(sub_path, f"cardiac_mri_cine_{subject}.nii.gz") 
        cine = os.path.join(sub_path, f"cardiac_mri_cine_{subject}.nii.gz") 

        data.append({"cine":cine, "dense":dense})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class CustomBase(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data = get_brats_dataset(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # import nibabel as nib
        # demo_dense = self.data[i]['dense']
        # source_img = demo_dense[0,...].cpu().numpy()
    
        # # Remove singleton dimensions and ensure correct orientation
        # tensor_numpy = np.squeeze(source_img)
        
        # # NiBabel expects data in (x, y, z) order, so we might need to transpose
        # # Assuming the tensor is in (depth, height, width) order:
        # #tensor_numpy = np.transpose(tensor_numpy, (2, 1, 0))
        
        # # Create a NIfTI image
        # nifti_image = nib.Nifti1Image(tensor_numpy, affine=np.eye(4))
        # nib.save(nifti_image, 'dense.nii.gz')
        return self.data[i]


class CustomTrain(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)


class CustomTest(CustomBase):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path)