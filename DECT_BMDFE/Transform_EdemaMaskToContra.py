import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time
from scipy.optimize import minimize

def FindLabels(patient_id):
    label_mat = np.loadtxt(open("SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4))
    patient_num = int(patient_id[len(patient_id)-4:])

    f_i = label_mat[patient_num-1,0]
    t_i = label_mat[patient_num-1,1]
    f_c = label_mat[patient_num-1,2]
    t_c = label_mat[patient_num-1,3]

    return f_i,t_i,f_c,t_c

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))
    patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    # print(side_num)
    return side_num



def TransformContra(participant_id,bone_id,filePath,injured_side):
    #Crop flipped filtered edema image and transform to align contralateral bone with injured side:
    GSI_fnm = 'GSI_Baseline'
    img_GSI = sitk.ReadImage(filePath+'/'+GSI_fnm+'.mha',sitk.sitkFloat32)
    edema_mask_fnm = 'edema_Thres_final_doublethres'
    edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'_'+bone_id+'.mha', sitk.sitkFloat32)
    edema_mask = edema_mask*100

    #resample mask to isotropic voxel size for FEA:
    resample_injured = sitk.ResampleImageFilter()
    resample_injured.SetReferenceImage(edema_mask)
    new_spacing = [0.625, 0.625, 0.625]
    resample_injured.SetOutputSpacing(new_spacing)
    resample_injured.SetInterpolator(sitk.sitkLinear)
    edema_mask_resampled = resample_injured.Execute(edema_mask)
    edema_mask_resampled = edema_mask_resampled>50
    sitk.WriteImage(edema_mask_resampled, filePath+'/'+participant_id+'_'+bone_id+'_edema_injured_625.mha',True)

    crop_DE = sitk.CropImageFilter()

    flipped_size = img_GSI.GetSize()

    crop_contra = sitk.CropImageFilter()
    if injured_side == 'right':
        crop_contra.SetLowerBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
        crop_contra.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'left':
        crop_contra.SetLowerBoundaryCropSize([0,0,0])
        crop_contra.SetUpperBoundaryCropSize([np.int(flipped_size[0]/2),0,0])

    img_contra = crop_contra.Execute(img_GSI)
    img_contra.SetOrigin([0,0,0])

    tfm_name = 'rigid_registration_contra_'+bone_id
    contra_tfm = sitk.ReadTransform(filePath+'/'+tfm_name+'.tfm')
    mask_tfm = contra_tfm.GetInverse()

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img_contra)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(mask_tfm)
    edema_mask_registered = resample.Execute(edema_mask)


    img_edema = edema_mask_registered


    img_edema.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
    img_edema.SetOrigin((0,0,0))


    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img_contra)
    tfm = sitk.CenteredTransformInitializer(img_contra,
                                            img_edema,
                                            sitk.VersorRigid3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(tfm)
    img_flipped = resample.Execute(img_edema)

    sitk.WriteImage(img_flipped>50, filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_Contra.mha',True)

    #resample  to isotropic voxel size for FEA:
    resample_contra = sitk.ResampleImageFilter()
    resample_contra.SetReferenceImage(img_flipped)
    new_spacing = [0.625, 0.625, 0.625]
    resample_contra.SetOutputSpacing(new_spacing)
    resample_contra.SetInterpolator(sitk.sitkLinear)
    img_flipped_resample = resample_contra.Execute(img_flipped)
    sitk.WriteImage(img_flipped_resample>50, filePath+'/'+participant_id+'_'+bone_id+'_edema_contra_625.mha',True)



def CropAndResampleGSI(participant_id,bone_id,filePath,injured_side):
    HA_fnm = 'GSI_Baseline'
    img_HA = sitk.ReadImage(filePath+'/'+HA_fnm+'.mha',sitk.sitkFloat32)

    img_size = img_HA.GetSize()
    crop_DE = sitk.CropImageFilter()
    if injured_side == 'left':
        crop_DE.SetLowerBoundaryCropSize([np.int(img_size[0]/2),0,0])
        crop_DE.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'right':
        crop_DE.SetLowerBoundaryCropSize([0,0,0])
        crop_DE.SetUpperBoundaryCropSize([np.int(img_size[0]/2),0,0])

    img_HA_injured = crop_DE.Execute(img_HA)
    img_HA_injured.SetOrigin([0,0,0])
    sitk.WriteImage(img_HA_injured,filePath+'/'+participant_id+'_'+HA_fnm+'_injured.mha',True)

    #resample images to isotropic voxel size for FEA:
    resample_injured = sitk.ResampleImageFilter()
    resample_injured.SetReferenceImage(img_HA_injured)
    new_spacing = [0.625, 0.625, 0.625]
    resample_injured.SetOutputSpacing(new_spacing)
    resample_injured.SetInterpolator(sitk.sitkLinear)
    sitk.WriteImage(resample_injured.Execute(img_HA_injured), filePath+'/'+participant_id+'_'+HA_fnm+'_injured_625.mha',True)

    crop_contra = sitk.CropImageFilter()
    if injured_side == 'left':
        crop_contra.SetLowerBoundaryCropSize([0,0,0])
        crop_contra.SetUpperBoundaryCropSize([np.int(img_size[0]/2),0,0])
    elif injured_side == 'right':
        crop_contra.SetLowerBoundaryCropSize([np.int(img_size[0]/2),0,0])
        crop_contra.SetUpperBoundaryCropSize([0,0,0])

    img_HA_contra = crop_contra.Execute(img_HA)
    img_HA_contra.SetOrigin([0,0,0])
    sitk.WriteImage(img_HA_contra,filePath+'/'+participant_id+'_'+HA_fnm+'_contra.mha',True)

    #resample images to isotropic voxel size for FEA:
    resample_contra = sitk.ResampleImageFilter()
    resample_contra.SetReferenceImage(img_HA_contra)
    new_spacing = [0.625, 0.625, 0.625]
    resample_contra.SetOutputSpacing(new_spacing)
    resample_contra.SetInterpolator(sitk.sitkLinear)
    sitk.WriteImage(resample_contra.Execute(img_HA_contra), filePath+'/'+participant_id+'_'+HA_fnm+'_contra_625.mha',True)



def main():
    # Set up description
    description='''Function to transform the edema mask from the injured knee to the contralateral.

    This program will read in a bilateral knee CT (GSI image), crop the image to isolate the injured and contralateral knees, and resample to isotropic voxel size (for FE) at 0.625 mm
    It also reads in the edema mask image (generated using Segment_Edema.py function), and transforms and mirrors it into the contralateral knee.
    The resulting file ([participant_id]_Tibia_Edema_Mask_Contra) will match the same VOI of the edema region on the contralateral knee to use as a matched, non-edema control
    Finally, this script will create edema masks (both in the injured and control knee) that are resampled to 0.625 mm isotropic voxel size for use in FEA


    Necessary files = GSI image, segmented edema mask, tfm file from contralateral registration


'''


    #Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("BoneID",
                        type=str,
                        help="The bone ID (Femur or Tibia)")

    args = parser.parse_args()

    filePath = args.filePath
    participant_id = args.participant_id
    bone_id = args.BoneID


    side_num = FindInjuredSide(participant_id)
    if side_num == 0:
        injured_side = 'left'
    elif side_num == 1:
        injured_side = 'right'


    TransformContra(participant_id,bone_id,filePath,injured_side)
    CropAndResampleGSI(participant_id,bone_id,filePath,injured_side)


if __name__ == '__main__':
    main()
