import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
# import ThresEdemaFcn_3mat_v2 as thres_fcn
import time
from scipy.optimize import minimize

def FindLabels(patient_id):
    label_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4))
    patient_num = int(patient_id[len(patient_id)-4:])

    f_i = label_mat[patient_num-1,0]
    t_i = label_mat[patient_num-1,1]
    f_c = label_mat[patient_num-1,2]
    t_c = label_mat[patient_num-1,3]

    return f_i,t_i,f_c,t_c

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))
    patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    # print(side_num)
    return side_num

def FindMRIThreshold(patient_id):
    PeakIntensity_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(5,6))
    patient_num = int(patient_id[len(patient_id)-4:])

    PeakIntensity = PeakIntensity_mat[patient_num-1,1]
    MRI_thres = PeakIntensity*0.2
    # print(MRI_thres)
    return MRI_thres


def TransformToXCT(participant_id,bone_id,filePath,injured_side):
    #Crop flipped filtered edema image and transform to align contralateral bone with injured side:
    gsi_fnm = participant_id+'_injured_calibrated_GSI'
    gsi = sitk.ReadImage(filePath+'/'+gsi_fnm+'.mha',sitk.sitkFloat32)
    gsi.SetOrigin((0,0,0))
    # edema_mask_fnm = participant_id+'_edema_NLMFiltered_cc2_doublethres'
    edema_mask_fnm = participant_id+'_largestcube'
    edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'.mha')
    # edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'_'+bone_id+'.mha')
    edema_mask = edema_mask*100
    edema_mask.SetOrigin((0,0,0))

    if injured_side == 'left':
        xct_fnm = participant_id + '_TL_T'
    elif injured_side == 'right':
        xct_fnm = participant_id + '_TR_T'

    xct = sitk.ReadImage(filePath+'/'+xct_fnm+'.mha',sitk.sitkFloat32)
    xct.SetDirection([1, 0, 0, 0, -1, 0, 0, 0, -1])
    xct.SetOrigin((0,0,0))
    # sitk.WriteImage(xct,filePath+'/'+xct_fnm+'_flipped.mha',True)
    sitk.WriteImage(xct,filePath+'/'+xct_fnm+'_flipped.mha')

    #resample mask to isotropic voxel size for FEA:
    # resample_injured = sitk.ResampleImageFilter()
    # resample_injured.SetReferenceImage(edema_mask)
    # new_spacing = [0.625, 0.625, 0.625]
    # resample_injured.SetOutputSpacing(new_spacing)
    # resample_injured.SetInterpolator(sitk.sitkLinear)
    # edema_mask_resampled = resample_injured.Execute(edema_mask)
    # edema_mask_resampled = edema_mask_resampled>50
    # sitk.WriteImage(edema_mask_resampled, filePath+'/'+participant_id+'_'+bone_id+'_edema_injured_625.mha',True)

    # crop_DE = sitk.CropImageFilter()

    # flipped_size = img_flipped.GetSize()
    # if injured_side == 'left':
    #     crop_DE.SetLowerBoundaryCropSize([0,0,0])
    #     crop_DE.SetUpperBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
    # elif injured_side == 'right':
    #     crop_DE.SetLowerBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
    #     crop_DE.SetUpperBoundaryCropSize([0,0,0])

    # img_flipped = crop_DE.Execute(img_flipped)
    # sitk.WriteImage(img_flipped,filePath+'/'+flipped_fnm+'_cropped.mha',True)
    # img_flipped.SetOrigin([0,0,0])

    tfm_name = xct_fnm+'_s4_XCT-DECT_registration'
    XCT_tfm = sitk.ReadTransform(filePath+'/'+tfm_name+'.tfm')
    DE_tfm = XCT_tfm.GetInverse()
    # mask_tfm = sitk.VersorRigid3DTransform()

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(xct)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(DE_tfm)
    edema_mask_registered = resample.Execute(edema_mask)
    gsi_registered = resample.Execute(gsi)

    # edema_mask_registered = edema_mask_registered > 50

    # sitk.WriteImage(edema_mask_registered>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_RegisterXCT.mha',True)
    sitk.WriteImage(edema_mask_registered>10,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Cube_RegisterXCT_fullres.mha',True)
    # sitk.WriteImage(gsi_registered,filePath+'/'+gsi_fnm+'_RegisterXCT_fullres.mha',True)

    #resample  to isotropic voxel size for FEA:
    # resample2 = sitk.ResampleImageFilter()
    # resample2.SetReferenceImage(xct)
    # xct_spacing = xct.GetSpacing()
    # xct_size = xct.GetSize()
    # new_spacing = [0.625, 0.625, 0.625]
    # new_size = [int(xct_size[0]*(xct_spacing[0]/0.625)),int(xct_size[1]*(xct_spacing[1]/0.625)),int(xct_size[2]*(xct_spacing[2]/0.625))]
    # resample2.SetOutputSpacing(new_spacing)
    # resample2.SetSize(new_size)
    # resample2.SetInterpolator(sitk.sitkLinear)
    # resample2.SetTransform(DE_tfm)
    # edema_mask_registered_2 = resample2.Execute(edema_mask)
    # gsi_registered2 = resample2.Execute(gsi)
    #
    # # sitk.WriteImage(edema_mask_registered_2>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_RegisterXCT_625.mha',True)
    # sitk.WriteImage(edema_mask_registered_2>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Cube_RegisterXCT_625.mha',True)
    # sitk.WriteImage(gsi_registered2,filePath+'/'+gsi_fnm+'_RegisterXCT_625.mha',True)



#Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("filePath",
                    type=str,
                    help="The filepath")
parser.add_argument("participantID",
                    type=str,
                    help="The bone ID (Femur or Tibia)")

args = parser.parse_args()

filePath = args.filePath
participant_id = args.participantID
# threshold_value = args.MR_Threshold
# a = os.path.split(filePath)
# b = len(a)
# participant_id = a[b-1]
# participant_id = 'SALTACII_'+participant_id
bone_id = 'Tibia'

side_num = FindInjuredSide(participant_id)
if side_num == 0:
    injured_side = 'left'
elif side_num == 1:
    injured_side = 'right'

# side_num = FindInjuredSide(participant_id)
# if side_num == 0:
#     injured_side = 'left'
# elif side_num == 1:
#     injured_side = 'right'
#

TransformToXCT(participant_id,bone_id,filePath,injured_side)
# CropAndResampleGSI(participant_id,bone_id,filePath,injured_side)
