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

def ResampleIsotropic(img,newsp):
    resample2 = sitk.ResampleImageFilter()
    resample2.SetReferenceImage(img)
    img_spacing = img.GetSpacing()
    img_size = img.GetSize()
    new_spacing = [newsp, newsp, newsp]
    new_size = [int(img_size[0]*(img_spacing[0]/newsp)),int(img_size[1]*(img_spacing[1]/newsp)),int(img_size[2]*(img_spacing[2]/newsp))]
    resample2.SetOutputSpacing(new_spacing)
    resample2.SetSize(new_size)
    resample2.SetInterpolator(sitk.sitkLinear)
    resample2.SetTransform(sitk.Euler3DTransform())
    img_resample = resample2.Execute(img)
    return img_resample

def IsolateDECTCubeVOI(particpant_id,bone_id,filePath,injured_side,box_offset,side_len):
    gsi_fnm = participant_id+'_injured_calibrated_GSI'
    gsi = sitk.ReadImage(filePath+'/'+gsi_fnm+'.mha',sitk.sitkFloat32)
    gsi.SetOrigin((0,0,0))
    gsi_org = gsi
    gsi = ResampleIsotropic(gsi,0.625)

    calibrated_dect_fnm = participant_id+'_injured_calibrated_DECT'
    calibrated_dect = sitk.ReadImage(filePath+'/'+calibrated_dect_fnm+'.mha',sitk.sitkFloat32)
    calibrated_dect.SetOrigin((0,0,0))
    dect_org = calibrated_dect
    calibrated_dect = ResampleIsotropic(calibrated_dect,0.625)

    # edema_mask_fnm = participant_id+'_edema_NLMFiltered_cc2_doublethres'
    edema_mask_fnm = participant_id+'_largestcube'
    edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'.mha')
    # edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'_'+bone_id+'.mha')
    edema_mask.SetOrigin((0,0,0))

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)

    # side_len = 13 #length of cube side, in voxels on 0.625 mm DECT
    box_size = [side_len+1, side_len+1, side_len+1]
    print(box_size)

    gsi_sp = gsi.GetSpacing()
    print(gsi_sp)
    # box_offset = [168,246,209] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0008
    # box_offset = [143,262,167] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0015
    # new_offset = [int(box_offset[0]*gsi_sp/xct_sp[0]), int(box_offset[1]*gsi_sp/xct_sp[1]),int(box_offset[2]*gsi_sp/xct_sp[2])]
    new_offset = [box_offset[0]*gsi_sp[0]-gsi_sp[0]*.5, box_offset[1]*gsi_sp[1]-gsi_sp[1]*0.5,box_offset[2]*gsi_sp[2]-gsi_sp[2]*.5]
    # new_offset = [box_offset[0]*gsi_sp[0], box_offset[1]*gsi_sp[1],box_offset[2]*gsi_sp[2]]


    resample1 = sitk.ResampleImageFilter()
    resample1.SetReferenceImage(gsi)
    resample1.SetOutputSpacing(gsi_sp)
    resample1.SetSize(box_size)
    resample1.SetOutputOrigin(new_offset)
    resample1.SetInterpolator(sitk.sitkLinear)
    resample1.SetTransform(sitk.Euler3DTransform())
    gsi_cube = resample1.Execute(gsi_org)
    dect_cube = resample1.Execute(dect_org)

    # gsi_cube = gsi*cast.Execute(edema_mask)
    #
    # masksize = edema_mask.GetSize()
    # stats_mask = sitk.LabelShapeStatisticsImageFilter()
    # stats_mask.Execute(edema_mask)
    # bound_mask = stats_mask.GetBoundingBox(1)
    # crop_mask = sitk.CropImageFilter()
    # crop_mask.SetLowerBoundaryCropSize([bound_mask[0], bound_mask[1], bound_mask[2]])
    # crop_mask.SetUpperBoundaryCropSize([(masksize[0]-bound_mask[0]-bound_mask[3]), (masksize[1]-bound_mask[1]-bound_mask[4]), (masksize[2]-bound_mask[2]-bound_mask[5])])
    # gsi_cube = crop_mask.Execute(gsi_cube)
    # gsi_cube.SetOrigin((0,0,0))

    sitk.WriteImage(gsi_cube,filePath+'/'+participant_id+'_CalibratedGSI_Cube_injured_625.mha',True)
    sitk.WriteImage(dect_cube,filePath+'/'+participant_id+'_CalibratedDECT_Cube_injured_625.mha',True)

def TransformToDECT(participant_id,bone_id,filePath,injured_side,box_offset,side_len):
    gsi_fnm = participant_id+'_injured_calibrated_GSI'
    gsi = sitk.ReadImage(filePath+'/'+gsi_fnm+'.mha',sitk.sitkFloat32)
    gsi.SetOrigin((0,0,0))
    gsi_org = gsi
    gsi_org_size = gsi_org.GetSize()
    gsi_org_sp = gsi_org.GetSpacing()
    gsi = ResampleIsotropic(gsi,0.625)

    sitk.WriteImage(gsi,filePath+'/'+gsi_fnm+'_625.mha',True)

    # edema_mask_fnm = participant_id+'_edema_NLMFiltered_cc2_doublethres'
    edema_mask_fnm = participant_id+'_largestcube'
    edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'.mha')
    # edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'_'+bone_id+'.mha')
    edema_mask.SetOrigin((0,0,0))


    if injured_side == 'left':
        xct_fnm = participant_id + '_TL_T'
    elif injured_side == 'right':
        xct_fnm = participant_id + '_TR_T'
    xct = sitk.ReadImage(filePath+'/'+xct_fnm+'.mha',sitk.sitkFloat32)
    xct.SetDirection([1, 0, 0, 0, -1, 0, 0, 0, -1])
    xct.SetOrigin((0,0,0))
    # sitk.WriteImage(xct,filePath+'/'+xct_fnm+'_flipped.mha')
    xct_sp = xct.GetSpacing()
    xct_size = xct.GetSize()
    # print(xct_size)

    # crop_xct = sitk.CropImageFilter()
    # # crop_xct.SetLowerBoundaryCropSize([np.int(xct_size[0]/2),480,0])
    # # crop_xct.SetUpperBoundaryCropSize([200,300,np.int(xct_size[2]/2)])
    # crop_xct.SetLowerBoundaryCropSize([np.int(xct_size[0]/2),480,0])
    # crop_xct.SetUpperBoundaryCropSize([0,0,np.int(xct_size[2]/2)])
    # xct = crop_xct.Execute(xct)
    # print(xct.GetSize())

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)

    # resample_mask = sitk.ResampleImageFilter()
    # resample_mask.SetReferenceImage(gsi)
    # resample_mask.SetOutputSpacing(xct_sp)
    # resample_mask.SetSize([int(gsi_org_size[0]*gsi_org_sp[0]/xct_sp[0]), int(gsi_org_size[1]*gsi_org_sp[1]/xct_sp[1]), int(gsi_org_size[2]*gsi_org_sp[2]/xct_sp[2])])
    #
    # resample_mask.SetInterpolator(sitk.sitkLinear)
    # resample_mask.SetTransform(sitk.VersorRigid3DTransform())
    # edema_mask = resample_mask.Execute(cast.Execute(edema_mask))
    # edema_mask = edema_mask>0.5

    # sitk.WriteImage(edema_mask,filePath+'/'+participant_id+'_cubemask_XCT.mha',True)

    tfm_name = xct_fnm+'_s4_XCT-DECT_registration'
    XCT_tfm = sitk.ReadTransform(filePath+'/'+tfm_name+'.tfm')

    side_len = side_len + 1 #length of cube side, in voxels on 0.625 mm DECT
    gsi_sp = 0.625
    box_size = [int(round((side_len)*gsi_sp/xct_sp[0])), int(round((side_len)*gsi_sp/xct_sp[0])), int(round((side_len)*gsi_sp/xct_sp[0]))]
    print(box_size)

    # box_offset = [168,246,209] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0008
    # box_offset[2] = box_offset[2]-1 #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0015
    # new_offset = [int(box_offset[0]*gsi_sp/xct_sp[0]), int(box_offset[1]*gsi_sp/xct_sp[1]),int(box_offset[2]*gsi_sp/xct_sp[2])]
    # new_offset = [box_offset[0]*gsi_sp, box_offset[1]*gsi_sp,box_offset[2]*gsi_sp]
    new_offset = [box_offset[0]*gsi_sp-gsi_sp*.5, box_offset[1]*gsi_sp-gsi_sp*0.5,box_offset[2]*gsi_sp-gsi_sp*.5]


    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(gsi_org)
    resample.SetOutputSpacing(xct_sp)
    resample.SetOutputOrigin(new_offset)
    resample.SetSize(box_size)
    # resample.SetSize([int(gsi_org_size[0]*gsi_org_sp[0]/xct_sp[0]), int(gsi_org_size[1]*gsi_org_sp[1]/xct_sp[1]), int(gsi_org_size[2]*gsi_org_sp[2]/xct_sp[2])])

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(XCT_tfm)
    xct_registered = resample.Execute(xct)

    # sitk.WriteImage(xct_registered,filePath+'/'+participant_id+'_XCT_registered.mha',True)

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)

    xct_cube = xct_registered
    # xct_cube = xct_registered*cast.Execute(edema_mask)

    # masksize = edema_mask.GetSize()
    # stats_mask = sitk.LabelShapeStatisticsImageFilter()
    # stats_mask.Execute(edema_mask)
    # bound_mask = stats_mask.GetBoundingBox(1)
    # crop_mask = sitk.CropImageFilter()
    # crop_mask.SetLowerBoundaryCropSize([bound_mask[0], bound_mask[1], bound_mask[2]])
    # crop_mask.SetUpperBoundaryCropSize([(masksize[0]-bound_mask[0]-bound_mask[3]), (masksize[1]-bound_mask[1]-bound_mask[4]), (masksize[2]-bound_mask[2]-bound_mask[5])])
    # xct_cube = crop_mask.Execute(xct_cube)
    # xct_cube.SetOrigin((0,0,0))

    sitk.WriteImage(xct_cube,filePath+'/'+participant_id+'_RegisteredXCT_Cube_injured.mha',True)


# def TransformToXCT(participant_id,bone_id,filePath,injured_side):
#     #Crop flipped filtered edema image and transform to align contralateral bone with injured side:
#     gsi_fnm = participant_id+'_GSI_Baseline_injured'
#     gsi = sitk.ReadImage(filePath+'/'+gsi_fnm+'.mha',sitk.sitkFloat32)
#     gsi.SetOrigin((0,0,0))
#     # edema_mask_fnm = participant_id+'_edema_NLMFiltered_cc2_doublethres'
#     edema_mask_fnm = participant_id+'_largestcube'
#     edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'.mha')
#     # edema_mask = sitk.ReadImage(filePath+'/'+edema_mask_fnm+'_'+bone_id+'.mha')
#     edema_mask = edema_mask*100
#     edema_mask.SetOrigin((0,0,0))
#     xct_fnm = participant_id+'_TL_T_s4'
#     xct = sitk.ReadImage(filePath+'/'+xct_fnm+'.mha',sitk.sitkFloat32)
#     xct.SetDirection([1, 0, 0, 0, -1, 0, 0, 0, -1])
#     xct.SetOrigin((0,0,0))
#     sitk.WriteImage(xct,filePath+'/'+xct_fnm+'_flipped.mha')
#
#     #resample mask to isotropic voxel size for FEA:
#     # resample_injured = sitk.ResampleImageFilter()
#     # resample_injured.SetReferenceImage(edema_mask)
#     # new_spacing = [0.625, 0.625, 0.625]
#     # resample_injured.SetOutputSpacing(new_spacing)
#     # resample_injured.SetInterpolator(sitk.sitkLinear)
#     # edema_mask_resampled = resample_injured.Execute(edema_mask)
#     # edema_mask_resampled = edema_mask_resampled>50
#     # sitk.WriteImage(edema_mask_resampled, filePath+'/'+participant_id+'_'+bone_id+'_edema_injured_625.mha',True)
#
#     # crop_DE = sitk.CropImageFilter()
#
#     # flipped_size = img_flipped.GetSize()
#     # if injured_side == 'left':
#     #     crop_DE.SetLowerBoundaryCropSize([0,0,0])
#     #     crop_DE.SetUpperBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
#     # elif injured_side == 'right':
#     #     crop_DE.SetLowerBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
#     #     crop_DE.SetUpperBoundaryCropSize([0,0,0])
#
#     # img_flipped = crop_DE.Execute(img_flipped)
#     # sitk.WriteImage(img_flipped,filePath+'/'+flipped_fnm+'_cropped.mha',True)
#     # img_flipped.SetOrigin([0,0,0])
#
#     tfm_name = participant_id+'_XCT-DECT_registration'
#     XCT_tfm = sitk.ReadTransform(filePath+'/'+tfm_name+'.tfm')
#     DE_tfm = XCT_tfm.GetInverse()
#     # mask_tfm = sitk.VersorRigid3DTransform()
#
#     resample = sitk.ResampleImageFilter()
#     resample.SetReferenceImage(xct)
#
#     resample.SetInterpolator(sitk.sitkLinear)
#     resample.SetTransform(DE_tfm)
#     edema_mask_registered = resample.Execute(edema_mask)
#     gsi_registered = resample.Execute(gsi)
#
#     # edema_mask_registered = edema_mask_registered > 50
#
#     # sitk.WriteImage(edema_mask_registered>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_RegisterXCT.mha',True)
#     sitk.WriteImage(edema_mask_registered>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Cube_RegisterXCT.mha',True)
#     sitk.WriteImage(gsi_registered,filePath+'/'+gsi_fnm+'_RegisterXCT.mha',True)
#
#     #resample  to isotropic voxel size for FEA:
#     resample2 = sitk.ResampleImageFilter()
#     resample2.SetReferenceImage(xct)
#     xct_spacing = xct.GetSpacing()
#     xct_size = xct.GetSize()
#     new_spacing = [0.625, 0.625, 0.625]
#     new_size = [int(xct_size[0]*(xct_spacing[0]/0.625)),int(xct_size[1]*(xct_spacing[1]/0.625)),int(xct_size[2]*(xct_spacing[2]/0.625))]
#     resample2.SetOutputSpacing(new_spacing)
#     resample2.SetSize(new_size)
#     resample2.SetInterpolator(sitk.sitkLinear)
#     resample2.SetTransform(DE_tfm)
#     edema_mask_registered_2 = resample2.Execute(edema_mask)
#     gsi_registered2 = resample2.Execute(gsi)
#
#     # sitk.WriteImage(edema_mask_registered_2>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Mask_RegisterXCT_625.mha',True)
#     sitk.WriteImage(edema_mask_registered_2>50,filePath+'/'+participant_id+'_'+bone_id+'_Edema_Cube_RegisterXCT_625.mha',True)
#     sitk.WriteImage(gsi_registered2,filePath+'/'+gsi_fnm+'_RegisterXCT_625.mha',True)
#


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

#Find cube offset and dimensions
cube_mask_fnm = participant_id+'_largestcube'
cube_mask = sitk.ReadImage(filePath+'/'+cube_mask_fnm+'.mha')
cube_mask.SetOrigin((0,0,0))

stats_mask = sitk.LabelShapeStatisticsImageFilter()
stats_mask.Execute(sitk.BinaryThreshold(cube_mask*2, lowerThreshold=1, upperThreshold=2, insideValue=2, outsideValue=0))
# stats_mask.Execute(sitk.BinaryThreshold(mask, lowerThreshold=0.5, upperThreshold=2, insideValue=2, outsideValue=0))
# stats_mask.Execute(mask)
bound_mask = stats_mask.GetBoundingBox(2)
print(bound_mask)

crop_mask = sitk.CropImageFilter()
box_offset = [bound_mask[0], bound_mask[1], bound_mask[2]]
# print(box_offset)
side_len = bound_mask[3]
# print(side_len)
# crop_mask.SetLowerBoundaryCropSize([bound_mask[0], bound_mask[1], bound_mask[2]])
# crop_mask.SetUpperBoundaryCropSize([(masksize[0]-bound_mask[0]-bound_mask[3]), (masksize[1]-bound_mask[1]-bound_mask[4]), (masksize[2]-bound_mask[2]-bound_mask[5])])

# masked_img = crop_mask.Execute(masked_img)


# box_offset = [175,264,206] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0005
# box_offset = [68,232,179] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0007
# box_offset = [168,246,209] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0008
# box_offset = [69,267,193] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0009
# box_offset = [143,262,167] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0015
# box_offset = [165,231,187] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0018
# box_offset = [65,240,199] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0002
# box_offset = [189,286,207] #offset of cube, in voxels, on 0.625 mm DECT SALTACII_0004


# side_len = 13 #length of cube side, in voxels on 0.625 mm DECT, SALTACII_0005, SALTACII_0008, SALTACII_0015
# side_len = 9 #length of cube side, in voxels on 0.625 mm DECT, SALTACII_0007, SALTACII_0009, SALTACII_0002, SALTACII_0004
# side_len = 11 #length of cube side, in voxels on 0.625 mm DECT, SALTACII_0018

IsolateDECTCubeVOI(participant_id,bone_id,filePath,injured_side,box_offset,side_len)
TransformToDECT(participant_id,bone_id,filePath,injured_side,box_offset,side_len)
# CropAndResampleGSI(participant_id,bone_id,filePath,injured_side)
