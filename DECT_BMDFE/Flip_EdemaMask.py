from __future__ import print_function
from functools import reduce


import SimpleITK as sitk
import sys
import os
import numpy as np
import pandas as pd
import argparse

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))
    patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    # print(side_num)
    return side_num


parser = argparse.ArgumentParser()
parser.add_argument("filePath",
                    type=str,
                    help="The filepath")
parser.add_argument("BoneID",
                    type=str,
                    help="The bone ID (Femur or Tibia)")

args = parser.parse_args()
filePath = args.filePath
bone_id = args.BoneID
# threshold_value = args.MR_Threshold
a = os.path.split(filePath)
b = len(a)
participant_id = a[b-1]
participant_id = 'SALTACII_'+participant_id

# img_fnm = 'edema_NLMFiltered'
img_fnm = 'GSI_Baseline'
img = sitk.ReadImage(filePath+'/'+img_fnm+'.mha', sitk.sitkFloat32)
img_edema = sitk.ReadImage(filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_RegisterContra.mha', sitk.sitkFloat32)
img_edema = img_edema*100


# seg_fnm = participant_id+'_40keV_SEG'
# img_seg = sitk.ReadImage(filePath+'/'+seg_fnm+'.mha', sitk.sitkFloat32)
# seg_flipped = sitk.ReadImage(filePath+'/'+seg_fnm+'.mha', sitk.sitkFloat32)

flipped_size = img.GetSize()
side_num = FindInjuredSide(participant_id)
if side_num == 0:
    injured_side = 'left'
elif side_num == 1:
    injured_side = 'right'
img_size = img.GetSize()
crop_DE = sitk.CropImageFilter()
if injured_side == 'right':
    crop_DE.SetLowerBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
    crop_DE.SetUpperBoundaryCropSize([0,0,0])
elif injured_side == 'left':
    crop_DE.SetLowerBoundaryCropSize([0,0,0])
    crop_DE.SetUpperBoundaryCropSize([np.int(flipped_size[0]/2),0,0])

img_contra = crop_DE.Execute(img)
img_contra.SetOrigin([0,0,0])
# sitk.WriteImage(img_contra,filePath+'/'+img_fnm+'_contra_cropped.mha',True)

img_edema.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
img_edema.SetOrigin((0,0,0))
# seg_flipped.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
# seg_flipped.SetOrigin((0,0,0))

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(img_contra)
tfm = sitk.CenteredTransformInitializer(img_contra,
                                        img_edema,
                                        sitk.VersorRigid3DTransform(),
                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

resample.SetInterpolator(sitk.sitkLinear)
resample.SetTransform(tfm)
img_flipped = resample.Execute(img_edema)
# img_flipped = img_flipped > 1

sitk.WriteImage(img_flipped>50, filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_Contra.mha',True)

#resample  to isotropic voxel size for FEA:
resample_contra = sitk.ResampleImageFilter()
resample_contra.SetReferenceImage(img_flipped)
new_spacing = [0.625, 0.625, 0.625]
resample_contra.SetOutputSpacing(new_spacing)
resample_contra.SetInterpolator(sitk.sitkLinear)
img_flipped_resample = resample_contra.Execute(img_flipped)
sitk.WriteImage(img_flipped_resample>50, filePath+'/'+participant_id+'_'+bone_id+'_edema_contra_625.mha',True)
