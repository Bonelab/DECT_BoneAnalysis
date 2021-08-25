import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse
from bonelab.util.echo_arguments import echo_arguments


def Flip_SagToAx(filePath,participant_id,img_fnm):
    moving = sitk.ReadImage(filePath+'/'+img_fnm+'.mha', sitk.sitkFloat32)

    moving.SetOrigin([0,0,0])

    moving_size = moving.GetSize()
    moving_spacing = moving.GetSpacing()

    #Flip YZ plane:
    tfm1 = sitk.Euler3DTransform()
    tfm1.SetRotation(-1*np.pi/2,0,0)
    tfm1.SetTranslation((0,0,int(moving_size[1]*moving_spacing[1])))
    tfm_flipyz = tfm1.GetInverse()
    print(tfm_flipyz)


    resample1 = sitk.ResampleImageFilter()
    resample1.SetReferenceImage(moving)
    resample1.SetSize((moving_size[0], moving_size[2], moving_size[1]))
    resample1.SetOutputSpacing((moving_spacing[0], moving_spacing[2], moving_spacing[1]))
    resample1.SetInterpolator(sitk.sitkLinear)
    resample1.SetTransform(tfm_flipyz)
    mr_flipYZ = resample1.Execute(moving)

    #Flip XY plane:
    mr_flipYZ.SetOrigin([0,0,0])
    moving_size = mr_flipYZ.GetSize()
    moving_spacing = mr_flipYZ.GetSpacing()

    tfm2 = sitk.Euler3DTransform()
    tfm2.SetRotation(0,0,-np.pi/2)
    tfm2.SetTranslation((0,1*int(moving_size[0]*moving_spacing[0]),0))
    tfm_flipxy = tfm2.GetInverse()

    resample2 = sitk.ResampleImageFilter()
    resample2.SetReferenceImage(mr_flipYZ)
    resample2.SetSize((moving_size[1], moving_size[0], moving_size[2]))
    resample2.SetOutputSpacing((moving_spacing[1], moving_spacing[0], moving_spacing[2]))
    resample2.SetInterpolator(sitk.sitkLinear)
    resample2.SetTransform(tfm_flipxy)
    mr_flipYZ_flipXY = resample2.Execute(mr_flipYZ)

    sitk.WriteImage(mr_flipYZ_flipXY, filePath+'/'+participant_id+'_'+img_fnm+'_flipYZ_flipXY.mha',True)




def main():
    # Set up description
    description='''Function to flip coordinate system from sagittal to axial.

    This function is used as first step to register a sagittally acquired MRI to an axial CT scan, for instance in SALTACII or KneeBML study.
    Should be run for both proton density (PD) and T2-weighted fat-saturated (T2 FS) images prior to image registration steps.

    Input image = sagittal MRI
    Output image = sagittal MRI flipped into axial coordinate system


'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Flip_Coordinates_SagittalToAxial",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--img_fnm","-if",
                        default='Sag_PD',
                        type=str,
                        help="Filename for the simululated monoenergetic CT image")


    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Flip_Coordinates_SagittalToAxial', vars(args)))

    # Run program
    Flip_SagToAx(**vars(args))


if __name__ == '__main__':
    main()
