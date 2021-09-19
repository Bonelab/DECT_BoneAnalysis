import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse
from bonelab.util.echo_arguments import echo_arguments

def FindLabels(patient_id):
    label_mat = np.loadtxt(open("SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4))
    patient_num = int(patient_id[len(patient_id)-4:])
    patient_num = 37

    f_i = label_mat[patient_num-1,0]
    t_i = label_mat[patient_num-1,1]
    f_c = label_mat[patient_num-1,2]
    t_c = label_mat[patient_num-1,3]

    return f_i,t_i,f_c,t_c

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))
    patient_num = int(patient_id[len(patient_id)-4:])
    patient_num = 37

    side_num = side_mat[patient_num-1,1]
    print(side_num)
    return side_num

def transform_T2(filePath,participant_id,fixed_img,moving_img,PD_CT_tfmname,T2_PD_tfmname,bone_id):


    output_filePath = filePath

    pixelType = sitk.sitkFloat32

    f_i,t_i,f_c,t_c = FindLabels(participant_id)
    if bone_id == 'Femur':
        BONE_LABEL = f_i
    elif bone_id == 'Tibia':
        BONE_LABEL = t_i

    mrfnm = moving_img
    mr = sitk.ReadImage(filePath+'/'+mrfnm+'.mha', sitk.sitkFloat32)
    mr.SetOrigin((0,0,0))

    mr_spacing = mr.GetSpacing()



    #read in fixed image:
    defnm = fixed_img
    dect_cropped = sitk.ReadImage(filePath+'/'+defnm+'.mha', sitk.sitkFloat32)
    fixed_size = dect_cropped.GetSize()
    de_spacing = dect_cropped.GetSpacing()


    #read in tfm:
    tfm1_name = participant_id+'_'+T2_PD_tfmname+'_'+bone_id
    tfm1 = sitk.ReadTransform(filePath+'/'+tfm1_name+'.tfm')
    tfm2_name = participant_id+'_'+PD_CT_tfmname+'_'+bone_id
    tfm2 = sitk.ReadTransform(filePath+'/'+tfm2_name+'.tfm')


    tfm_combined = sitk.Transform(tfm1)
    tfm_combined.AddTransform(tfm2)



    #apply tfm to moving image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(dect_cropped)


    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(tfm_combined)
    sitk.WriteImage(resample.Execute(mr), output_filePath+'/'+mrfnm+'-registeredDECT.mha',True)

def main():
    # Set up description
    description='''Function to transform T2 FS MRI to align with DECT.

    This script will transform the T2 FS MRI to align with the DECT image.
    Getting this alignment to work is essentially a two-step process, using a T1 or PD-weighted MRI as a registration intermediate:
    The fluid-sensitive T2 FS MRI must have been previously registered to an MRI showing more of the bony anatomy (T1 or PD weighted) (Register_PDMRI-DECT_LandmarkInitialized.py)
    The T1/PD MRI must have been previously registered to the DECT image (Register_T2-PDMRI.py)

    In addition, if acquired in the sagittal plane (as for SALTACII and KneeBML studies) the T2 MRI should have been previously flipped into an axial coordinate system
    to match the CT (Flip_Coordinates_SagittalToAxial.py)

    This script will combine the two transformations (PD/T1-to-DECT and T2-to-PD/T1), and apply the result to transform the T2 FS image into CT image space

    Input files = Sagittal T2 MRI that has been transformed to axial, DECT image (cropped to include only injured knee), PD/T1-to-DECT tfm file, T2-to-PD/T1 tfm file.


'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Transform_T2MRI-DECT",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--fixed_img","-if",
                        default='40keV_cropped',
                        type=str,
                        help="Filename for the fixed image (cropped DECT)")
    parser.add_argument("--moving_img","-im",
                        default='Sag_T2_FS_flipYZ_flipXY',
                        type=str,
                        help="Filename for the moving image (T2 FS MRI)")
    parser.add_argument("--PD_CT_tfmname","-tfm1",
                        default='PD-CT_registration',
                        type=str,
                        help="Base filename for transformation file to transform PD to CT")
    parser.add_argument("--T2_PD_tfmname","-tfm2",
                        default='T2-PD_registration',
                        type=str,
                        help="Base filename for transformation file to transform T2 to PD")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")



    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Transform_T2MRI-DECT', vars(args)))

    # Run program
    transform_T2(**vars(args))


if __name__ == '__main__':
    main()
