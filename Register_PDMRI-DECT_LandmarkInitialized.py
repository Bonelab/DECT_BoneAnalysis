from __future__ import print_function
from functools import reduce


import SimpleITK as sitk
import sys
import os
import ManualInitialGuessLibrary_py3
import numpy as np
import pandas as pd
import argparse
from bonelab.util.echo_arguments import echo_arguments


def command_iteration(method) :
    print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))



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
    return side_num

def register_PD_CT(filePath,participant_id,CT_fnm,MRI_fnm,mask_fnm,bone_id):

    pixelType = sitk.sitkFloat32

    f_i,t_i,f_c,t_c = FindLabels(participant_id)
    if bone == 'Femur':
        BONE_LABEL = f_i
    elif bone == 'Tibia':
        BONE_LABEL = t_i


    side_num = FindInjuredSide(participant_id)
    if side_num == 0:
        injured_side = 'left'
    elif side_num == 1:
        injured_side = 'right'

    fixedfnm = CT_fnm
    fixed = sitk.ReadImage(filePath+'/'+fixedfnm+'.mha', sitk.sitkFloat32)
    fixed_type = 'CT'
    img_size = fixed.GetSize()

    movingfnm = MRI_fnm
    moving = sitk.ReadImage(filePath+'/'+movingfnm+'.mha', sitk.sitkFloat32)
    moving_type = 'MR'

    #Crop CT images to focus on injured side only:
    crop_DE = sitk.CropImageFilter()


    if injured_side == 'left':
        crop_DE.SetLowerBoundaryCropSize([np.int(img_size[0]/2),0,0])
        crop_DE.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'right':
        crop_DE.SetLowerBoundaryCropSize([0,0,0])
        crop_DE.SetUpperBoundaryCropSize([np.int(img_size[0]/2),0,0])


    fixed_cropped = crop_DE.Execute(fixed)
    fixed_cropped.SetOrigin((0,0,0))
    sitk.WriteImage(fixed_cropped, filePath+'/'+fixedfnm+'_cropped.mha',True)

    seg_img = sitk.ReadImage(filePath+'/'+mask_fnm+'.mha')
    seg_img = crop_DE.Execute(seg_img)
    seg_img.SetOrigin((0,0,0))
    sitk.WriteImage(seg_img, filePath+'/'+mask_fnm+'_cropped.mha',True)

    fixed_image_mask = seg_img==BONE_LABEL
    sitk.WriteImage(fixed_image_mask, filePath+'/'+mask_fnm+'_cropped_'+bone+'.mha',True)

    #create a mask for image registration that is focused on the cortical region only (with some dilation in both directions)
    #excludes most of marrow and external soft tissues
    radius = 5
    dilated_fixed_image_mask = sitk.BinaryDilate(
            fixed_image_mask,
            radius
            )
    eroded_fixed_image_mask = sitk.BinaryErode(
            fixed_image_mask,
            radius
            )

    fixed_image_mask = dilated_fixed_image_mask - eroded_fixed_image_mask


    sitk.WriteImage(fixed_image_mask, filePath+'/'+fixedfnm+'_Dilated_mask.mha',True)


    moving_cropped = moving
    moving_cropped.SetOrigin((0,0,0))


    #This initial transformation is just to get the MRI image in the same dimensions as DE-CT
    initial_transformation1 = sitk.CenteredTransformInitializer(fixed_cropped,
                                                      moving_cropped,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    print(initial_transformation1)
    resample1 = sitk.ResampleImageFilter()
    resample1.SetReferenceImage(fixed_cropped)

    resample1.SetInterpolator(sitk.sitkLinear)
    resample1.SetTransform(initial_transformation1)
    sitk.WriteImage(resample1.Execute(moving_cropped), filePath+'/'+movingfnm+'-centered_'+bone+'.mha',True)


    # Manually select landmarks, and generate transformation based on the landmarks
    fixed_initialpts, moving_initialpts = ManualInitialGuessLibrary_py3.GetPoints(filePath,fixedfnm+'_cropped',movingfnm+'-centered_'+bone,fixed_type,moving_type)

    d2 = {'Landmark1_fixed': fixed_initialpts[0],'Landmark2_fixed': fixed_initialpts[1], 'Landmark3_fixed': fixed_initialpts[2], 'Landmark4_fixed': fixed_initialpts[3], 'Landmark1_moving': moving_initialpts[0], 'Landmark2_moving': moving_initialpts[1], 'Landmark3_moving': moving_initialpts[2], 'Landmark4_moving': moving_initialpts[3]}
    df2 = pd.DataFrame(data=d2)
    df2.to_csv(filePath+'/'+fixedfnm+'_Landmarks_'+bone+'.csv')

    print('fixed_initialpts', fixed_initialpts)
    print('moving_initialpts', moving_initialpts)



    fixed_image_points_flat = [c for p in fixed_initialpts for c in p]
    moving_image_points_flat = [c for p in moving_initialpts for c in p]


    initial_transformation2 = sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(),
                                                                fixed_image_points_flat,
                                                                moving_image_points_flat)

    print(initial_transformation2)

    #Combine the 2 transformations
    initial_transformation = sitk.Transform(initial_transformation1)
    initial_transformation.AddTransform(initial_transformation2)

    resample2 = sitk.ResampleImageFilter()
    resample2.SetReferenceImage(fixed_cropped)

    resample2.SetInterpolator(sitk.sitkLinear)
    resample2.SetTransform(initial_transformation2)
    sitk.WriteImage(resample2.Execute(resample1.Execute(moving_cropped)), filePath+'/'+movingfnm+'-manualinitialguess_'+bone+'.mha',True)



    sitk.WriteTransform(initial_transformation, filePath+'/'+participant_id+'_T1-CT_manualinitialguess_'+bone+'.tfm')

    #landmark-initialized registration:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsPowell()
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transformation)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetMetricFixedMask(fixed_image_mask)

    def callback(R):
        i = R.GetOptimizerIteration()
        m = R.GetMetricValue()
        print(i,m)

    R.AddCommand(sitk.sitkIterationEvent, lambda: callback(R))


    outTx = R.Execute(sitk.Normalize(fixed_cropped), sitk.Normalize(moving_cropped))
    sitk.WriteTransform(outTx, filePath+'/'+participant_id+'_T1-CT_registration_'+bone+'.tfm')


    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))


    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_cropped)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(outTx)
    sitk.WriteImage(resample.Execute(moving_cropped), filePath+'/'+movingfnm+'-registered_'+bone+'.mha',True)

def main():
    # Set up description
    description='''Function to register left and right knees within one individual.

    This program will read in a bilateral knee CT (acquired using SALTACII protocol), and crop the image to separate the contralateral and injured knees (identified in commandline arguments)
    The contralateral knee will then be mirrored to match the injured knee, and image registration will be performed to align the mirrored contralateral and injured sides.
    Separate registrations should be performed for the femur & tibia.


    Input image = simulated monoenergetic image (e.g., at 40 keV), mask of major bones (generated through method of Krčah, Marcel, Gábor Székely, and Rémi Blanc, IEEE, 2011.)


'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Register_PDMRI-DECT_LandmarkInitialized",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--CT_fnm","-ctf",
                        default='40keV',
                        type=str,
                        help="Filename for the simululated monoenergetic CT image")
    parser.add_argument("--MRI_fnm","-mrf",
                        default='Sag_PD_flipYZ_flipXY',
                        type=str,
                        help="Filename for the PD or T1 weighted MR image (flipped into axial coordinates)")
    parser.add_argument("--mask_fnm","-m",
                        default='Calibrated_StandardSEQCT_SEG',
                        type=str,
                        help="Filename for the periosteal mask image of major bones")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")

    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('DECT_LandmarkInitialized', vars(args)))

    # Run program
    register_PD_CT(**vars(args))


if __name__ == '__main__':
    main()
