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

def register_T2_PD(filePath,participant_id,fixed_img,moving_img,mask_fnm,bone_id,injured_side):

    output_filePath = filePath

    pixelType = sitk.sitkFloat32

    #Read images:
    fixedfnm = participant_id+'_'+fixed_img
    fixed = sitk.ReadImage(filePath+'/'+fixedfnm+'.mha', sitk.sitkFloat32)
    fixed_size = fixed.GetSize()
    fixed_spacing = fixed.GetSpacing()

    movingfnm = participant_id+'_'+moving_img
    moving = sitk.ReadImage(filePath+'/'+movingfnm+'.mha', sitk.sitkFloat32)
    moving.SetOrigin([0,0,0])
    moving_size = moving.GetSize()
    moving_spacing = moving.GetSpacing()


    f_i,t_i,f_c,t_c = FindLabels(participant_id)
    if bone_id == 'Femur':
        BONE_LABEL = f_i
    elif bone_id == 'Tibia':
        BONE_LABEL = t_i

    ctmask_fnm = participant_id+'_'+mask_fnm
    ct_mask = sitk.ReadImage(filePath+'/'+ctmask_fnm+'.mha')

    #Crop mask to isolate injured knee
    img_size = ct_mask.GetSize()
    crop_img = sitk.CropImageFilter()

    if injured_side == 'left':
        crop_img.SetLowerBoundaryCropSize([np.int(img_size[0]/2),0,0])
        crop_img.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'right':
        crop_img.SetLowerBoundaryCropSize([0,0,0])
        crop_img.SetUpperBoundaryCropSize([np.int(img_size[0]/2),0,0])

    ct_mask = crop_img.Execute(ct_mask)
    ct_mask.SetOrigin((0,0,0))
    sitk.WriteImage(ct_mask,output_filePath+'/'+ctmask_fnm+'_injured.mha',True)
    ct_mask = ct_mask == BONE_LABEL

    #Transform periosteal mask image from CT image space to T1/PD image space (using Ti/PD-DECT transformation file):
    T1_CT_tfmname = participant_id+'_PD-CT_registration_'+bone_id
    T1toCT_tfm = sitk.ReadTransform(filePath+'/'+T1_CT_tfmname+'.tfm')
    T1toCT_tfm.SetInverse()

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)
    resample.SetTransform(T1toCT_tfm)
    resample.SetInterpolator(sitk.sitkLinear)
    T1_mask = resample.Execute(ct_mask)

    T1_mask = T1_mask >= 1
    sitk.WriteImage(T1_mask,output_filePath+'/'+participant_id+'_PD_mask.mha',True)

    #dilate mask image by 5 voxels to include a bit outside of periosteal surface
    radius = 5
    dilated_T1_mask = sitk.BinaryDilate(
            T1_mask,
            radius
            )


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsPowell()
    R.SetOptimizerScalesFromPhysicalShift()
    initial_transformation = sitk.CenteredTransformInitializer(fixed,
                                                     moving,
                                                     sitk.Euler3DTransform(),
                                                     sitk.CenteredTransformInitializerFilter.GEOMETRY)

    R.SetInitialTransform(initial_transformation)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricFixedMask(dilated_T1_mask)

    def callback(R):
        i = R.GetOptimizerIteration()
        m = R.GetMetricValue()
        print(i,m)

    R.AddCommand(sitk.sitkIterationEvent, lambda: callback(R))


    outTx = R.Execute(sitk.Normalize(fixed), sitk.Normalize(moving))
    sitk.WriteTransform(outTx, output_filePath+'/'+participant_id+'_T2-PD_registration_'+bone_id+'.tfm')

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(outTx)
    sitk.WriteImage(resample.Execute(moving), output_filePath+'/'+movingfnm+'-registeredPD_'+bone_id+'.mha',True)

def main():
    # Set up description
    description='''Function to register T2 FS to PD MRI.

    This script will perform image registration to ensure that the T2 FS MRI and proton density (or T1) MRI are aligned with each other.
    If registering the T2 FS MRI to DECT, this registration should be used as an intermediate step (first flip MRI coordinates from sagittal to axial, then register PD/T1 to DECT, then register T2 FS MRI to PD/T1, then combine the two registrations)
    before running this script, T1/PD MRI must already be registered to DECT, and the resulting transformation file must be in the same file location as input images.
    Separate registrations should be performed for the femur & tibia.


    Input images = T2 FS MRI (moving image), proton density (PD) or T1 MRI (fixed image), mask of major bones (generated through method of Krčah, Marcel, Gábor Székely, and Rémi Blanc, IEEE, 2011.)


'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Register_MRI",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--fixed_img","-if",
                        default='Sag_PD_flipYZ_flipXY',
                        type=str,
                        help="Filename for the fixed image (PD or T1 MRI)")
    parser.add_argument("--moving_img","-im",
                        default='Sag_T2_FS_flipYZ_flipXY',
                        type=str,
                        help="Filename for the moving image (T2 FS MRI)")
    parser.add_argument("--mask_fnm","-m",
                        default='40keV_SEG',
                        type=str,
                        help="Filename for the mask image of major bones")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")
    parser.add_argument("--injured_side","-i",
                        default='left',
                        type=str,
                        help="Side of injury (right or left)")


    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Register_MRI', vars(args)))

    # Run program
    register_T2_PD(**vars(args))


if __name__ == '__main__':
    main()
