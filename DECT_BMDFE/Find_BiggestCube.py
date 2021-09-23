import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))
    patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    return side_num

def ResampleIsotropic(img):
    resample2 = sitk.ResampleImageFilter()
    resample2.SetReferenceImage(img)
    img_spacing = img.GetSpacing()
    img_size = img.GetSize()
    new_spacing = [0.625, 0.625, 0.625]
    new_size = [int(img_size[0]*(img_spacing[0]/0.625)),int(img_size[1]*(img_spacing[1]/0.625)),int(img_size[2]*(img_spacing[2]/0.625))]
    resample2.SetOutputSpacing(new_spacing)
    resample2.SetSize(new_size)
    resample2.SetInterpolator(sitk.sitkLinear)
    resample2.SetTransform(sitk.Euler3DTransform())
    img_resample = resample2.Execute(img)
    return img_resample

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
# injured_side = 'left'

side_num = FindInjuredSide(participant_id)
if side_num == 0:
    injured_side = 'left'
elif side_num == 1:
    injured_side = 'right'

gsi_name = participant_id+'_GSI_Baseline_injured'
gsi = sitk.ReadImage(filePath+'/'+gsi_name+'.mha',sitk.sitkFloat32)
gsi = ResampleIsotropic(gsi)
sitk.WriteImage(gsi,filePath+'/'+gsi_name+'_625.mha',True)

mask_name = participant_id + '_edema_NLMFiltered_cc2_doublethres_Tibia'
edema_mask = sitk.ReadImage(filePath+'/'+mask_name+'.mha',sitk.sitkFloat32)
edema_mask = ResampleIsotropic(edema_mask)
edema_mask = edema_mask > 0.5
sitk.WriteImage(edema_mask,filePath+'/'+mask_name+'_625.mha',True)

if injured_side == 'left':
    xct_name = participant_id + '_TL_T_s4-registered'
elif injured_side == 'right':
    xct_name = participant_id + '_TR_T_s4-registered'

xct = sitk.ReadImage(filePath+'/'+xct_name+'.mha',sitk.sitkFloat32)
xct = ResampleIsotropic(xct)
sitk.WriteImage(xct,filePath+'/'+xct_name+'_625.mha',True)
xct_thres = xct > 0

edema_mask = edema_mask*xct_thres

sitk.WriteImage(edema_mask,filePath+'/'+participant_id+'_edema_mask_boundXCT.mha',True)

dilate_filter = sitk.BinaryDilateImageFilter()
# dilate_filter.SetKernelRadius(4) #for SALTACII_0009
dilate_filter.SetKernelRadius(1)
edema_mask = dilate_filter.Execute(edema_mask)

close_filter = sitk.BinaryFillholeImageFilter()
close_filter.SetForegroundValue(1)
edema_mask = close_filter.Execute(edema_mask)


# masksize = edema_mask.GetSize()
# stats_mask = sitk.LabelShapeStatisticsImageFilter()
# stats_mask.Execute(edema_mask)
# bound_mask = stats_mask.GetBoundingBox(1)
# crop_mask = sitk.CropImageFilter()
# crop_mask.SetLowerBoundaryCropSize([bound_mask[0], bound_mask[1], bound_mask[2]])
# crop_mask.SetUpperBoundaryCropSize([(masksize[0]-bound_mask[0]-bound_mask[3]), (masksize[1]-bound_mask[1]-bound_mask[4]), (masksize[2]-bound_mask[2]-bound_mask[5])])
# edema_mask = crop_mask.Execute(edema_mask)
#
# sitk.WriteImage(edema_mask, filePath+'/'+participant_id+'_test_closeholes.mha',True)


cast = sitk.CastImageFilter()
cast.SetOutputPixelType(sitk.sitkFloat32)

#compute distance transform
dt = sitk.SignedMaurerDistanceMapImageFilter()
dt.InsideIsPositiveOn()
#dt.UseImageSpacingOn()
dt.SquaredDistanceOff()
# dt.UseImageSpacingOn()
distancemap = dt.Execute(edema_mask)*cast.Execute(edema_mask)

sitk.WriteImage(distancemap,filePath+'/'+participant_id+'_distancemap.mha',True)

max_filter = sitk.MinimumMaximumImageFilter()
max_filter.Execute(distancemap)
max_value = max_filter.GetMaximum()
print(max_value)

edema_max = distancemap == max_value

cc_filter = sitk.ConnectedComponentImageFilter()
cl = cc_filter.Execute(edema_max)

cc_relabel = sitk.RelabelComponentImageFilter()
cc_relabel.SetMinimumObjectSize(1)
cl_relabeled = cc_relabel.Execute(cl)
max_filter = sitk.MinimumMaximumImageFilter()
max_filter.Execute(cl_relabeled)
max_size = max_filter.GetMaximum()
print(max_size)
# cl_relabeled_thres = cl_relabeled >=1
cl_relabeled_largest = cl_relabeled == max_size

# cl_relabeled_largest.SetPixel(72,237,183,0) #For SALTACII_0007 -- if more than one voxel at max thickness location, set extra voxel to zero

# cl_relabeled_largest.SetPixel(106,223,187,0) #For SALTACII_0013 -- need to adjust cube location to fit entirely within HR-pQCT
# cl_relabeled_largest.SetPixel(106,223,185,1) #For SALTACII_0013 -- need to adjust cube location to fit entirely within HR-pQCT

cl_relabeled_largest.SetPixel(87,203,212,0) #For SALTACII_0012 -- need to adjust cube location to fit entirely within HR-pQCT
cl_relabeled_largest.SetPixel(87,203,210,1) #For SALTACII_0012 -- need to adjust cube location to fit entirely within HR-pQCT


#create cube VOI
if max_value*np.sqrt(2)/2 < 4: # for FE, we need a min cube side length of 5 mm (so each side should be at least 8 voxels long)
    r = 4
else:
    r = int(round(max_value*np.sqrt(2)/2))

print(r)

bin_dil = sitk.BinaryDilateImageFilter()
bin_dil.SetKernelType(sitk.sitkBox)
bin_dil.SetKernelRadius(r)
square_voi = bin_dil.Execute(cl_relabeled_largest)

sitk.WriteImage(square_voi,filePath+'/'+participant_id+'_largestcube.mha',True)
