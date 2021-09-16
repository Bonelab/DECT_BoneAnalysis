from __future__ import print_function
from functools import reduce


import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse
from bonelab.util.echo_arguments import echo_arguments

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
    if patient_id == "SALTACII_0037":
        patient_num = 36
    elif patient_id == "SALTACII_0040":
        patient_num = 37
    else:
        patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    return side_num

def mirror_img(filePath,img_fnm,img,img_flipped):
    #Creates mirrored (flipped) images -- needed to allow contralateral knee to align with injured knee
    img_flipped.SetDirection([-1, 0, 0, 0, 1, 0, 0, 0, 1])
    img_flipped.SetOrigin((0,0,0))

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img)
    tfm = sitk.CenteredTransformInitializer(img,
                                            img_flipped,
                                            sitk.VersorRigid3DTransform(),
                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(tfm)
    img_mirrored = resample.Execute(img_flipped)
    sitk.WriteImage(img_mirrored, filePath+'/'+img_fnm+'_mirrored.mha',True)
    return img_mirrored


def register_contra(filePath,participant_id,img_fnm,mask_fnm,bone_id):
    # a = os.path.split(filePath)
    # b = len(a)
    # participant_id = a[b-1]
    output_filePath = filePath
    pixelType = sitk.sitkFloat32

    #Find mask labels for bone of interest (this assumes that labels are saved in a file called: SegLabels.csv)
    f_i,t_i,f_c,t_c = FindLabels(participant_id)
    if bone_id == 'Femur':
        bone_label = f_i
        contra_label = f_c
    elif bone_id == 'Tibia':
        bone_label = t_i
        contra_label = t_c

    side_num = FindInjuredSide(participant_id)
    if side_num == 0:
        injured_side = 'left'
    elif side_num == 1:
        injured_side = 'right'

    #Read images
    img = sitk.ReadImage(filePath+'/'+img_fnm+'.mha', sitk.sitkFloat32)
    img_flipped = sitk.ReadImage(filePath+'/'+img_fnm+'.mha', sitk.sitkFloat32)
    img_mask = sitk.ReadImage(filePath+'/'+mask_fnm+'.mha',sitk.sitkFloat32)
    mask_flipped = sitk.ReadImage(filePath+'/'+mask_fnm+'.mha', sitk.sitkFloat32)

    #Create mirrored images:
    img_flipped = mirror_img(filePath,img_fnm,img,img_flipped)
    mask_flipped = mirror_img(filePath,mask_fnm,img_mask,mask_flipped)

    img_mask = img_mask  == bone_label

    #dilate mask image by 5 voxels to include a bit outside of periosteal surface
    radius = 5
    dilated_mask = sitk.BinaryDilate(img_mask,radius)

    contra = img_flipped
    contra_mask = mask_flipped == contra_label

    #Crop images to isolate injured/contralateral knee
    img_size = img.GetSize()
    crop_img = sitk.CropImageFilter()

    if injured_side == 'left':
        crop_img.SetLowerBoundaryCropSize([np.int(img_size[0]/2),0,0])
        crop_img.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'right':
        crop_img.SetLowerBoundaryCropSize([0,0,0])
        crop_img.SetUpperBoundaryCropSize([np.int(img_size[0]/2),0,0])

    img_cropped = crop_img.Execute(img)
    dilated_mask = crop_img.Execute(dilated_mask)
    contra_cropped = crop_img.Execute(contra)
    img_mask = crop_img.Execute(img_mask)
    contra_mask = crop_img.Execute(contra_mask)

    img_cropped.SetOrigin((0,0,0))
    dilated_mask.SetOrigin((0,0,0))
    contra_cropped.SetOrigin((0,0,0))
    img_mask.SetOrigin((0,0,0))
    contra_mask.SetOrigin((0,0,0))

    sitk.WriteImage(img_cropped,output_filePath+'/'+img_fnm+'_injured_cropped.mha',True)
    sitk.WriteImage(contra_cropped,output_filePath+'/'+img_fnm+'_contra_cropped_mirrorred.mha',True)
    sitk.WriteImage(img_mask, output_filePath+'/'+'injured_mask_cropped.mha',True)
    sitk.WriteImage(contra_mask, output_filePath+'/'+'contra_mask_cropped_mirrored.mha',True)

    #Register the mirrored contralateral to the injured knee, using only data from the bone of interest (tibia or femur),
    #with registration initialization based on moments, and mutual information similarity metric
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    # R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsPowell()
    R.SetOptimizerScalesFromPhysicalShift()
    initial_transformation = sitk.CenteredTransformInitializer(img_mask,
                                                     contra_mask,
                                                     sitk.Euler3DTransform(),
                                                     sitk.CenteredTransformInitializerFilter.MOMENTS)


    resample0 = sitk.ResampleImageFilter()
    resample0.SetReferenceImage(img_cropped)
    resample0.SetInterpolator(sitk.sitkLinear)
    resample0.SetTransform(initial_transformation)
    sitk.WriteImage(resample0.Execute(contra_cropped), output_filePath+'/'+img_fnm+'_contra_initialguess_'+bone_id+'.mha',True)

    R.SetInitialTransform(initial_transformation)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetMetricFixedMask(dilated_mask)

    def callback(R):
        i = R.GetOptimizerIteration()
        m = R.GetMetricValue()
        print(i,m)

    R.AddCommand(sitk.sitkIterationEvent, lambda: callback(R))


    outTx = R.Execute(img_cropped,contra_cropped)
    sitk.WriteTransform(outTx, output_filePath+'/'+'rigid_registration_contra_'+bone_id+'.tfm')


    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img_cropped)

    # resample.SetInterpolator(sitk.sitkLinear)
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetTransform(outTx)
    contra_registered1 = resample.Execute(contra_cropped)
    contra_mask_registered1 = resample.Execute(contra_mask)
    sitk.WriteImage(contra_registered1, output_filePath+'/'+img_fnm+'_contra_rigidregistration_'+bone_id+'.mha',True)
    sitk.WriteImage(contra_mask_registered1, output_filePath+'/'+'contra_mask_rigidregistration_'+bone_id+'.mha',True)

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
        prog="Register_Injured-Contralateral",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--img_fnm","-if",
                        default='40keV',
                        type=str,
                        help="Filename for the simululated monoenergetic CT image")
    parser.add_argument("--mask_fnm","-m",
                        default='Calibrated_StandardSEQCT_SEG',
                        type=str,
                        help="Filename for the mask image of major bones")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")

    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Register_Injured-Contralateral', vars(args)))

    # Run program
    register_contra(**vars(args))


if __name__ == '__main__':
    main()
