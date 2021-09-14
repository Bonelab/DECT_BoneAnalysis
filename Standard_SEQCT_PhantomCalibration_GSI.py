import SimpleITK as sitk
import sys
import os
import numpy as np
import pandas as pd
import argparse
from bonelab.util.echo_arguments import echo_arguments



def SEQCT_Calibration(filePath,GSI_filename,HA_mask_fnm):
    a = os.path.split(filePath)
    b = len(a)
    participant_id = a[b-1]


    #read in images:
    mono_40_fnm = 'GSI_Baseline'
    # mono_40 = sitk.ReadImage(filePath+'/'+mono_40_fnm+'.mha', sitk.sitkFloat32)
    mono_40 = sitk.ReadImage(filePath+'/'+mono_40_fnm+'.mha', sitk.sitkFloat32)
    img_size = mono_40.GetSize()
    img_spacing = mono_40.GetSpacing()
    img_origin = mono_40.GetOrigin()

    side_num = FindInjuredSide(participant_id)
    if side_num == 0:
        injured_side = 'left'
    elif side_num == 1:
        injured_side = 'right'

    #Calculate average intensity in phantom and find line of best fit:
    HA800_label = 3
    HA400_label = 2
    HA100_label = 1

    HAphantom_mask = sitk.ReadImage(filePath+'/'+participant_id+'_HA_mask.mha')
    HA800_mask = HAphantom_mask == HA800_label
    HA400_mask = HAphantom_mask == HA400_label
    HA100_mask = HAphantom_mask == HA100_label

    HA800_phantom = sitk.Mask(mono_40,HA800_mask)
    HA400_phantom = sitk.Mask(mono_40,HA400_mask)
    HA100_phantom = sitk.Mask(mono_40,HA100_mask)


    HA800_array = sitk.GetArrayFromImage(HA800_phantom)
    HA800_mask_array = sitk.GetArrayFromImage(HA800_mask)
    HA400_array = sitk.GetArrayFromImage(HA400_phantom)
    HA400_mask_array = sitk.GetArrayFromImage(HA400_mask)
    HA100_array = sitk.GetArrayFromImage(HA100_phantom)
    HA100_mask_array = sitk.GetArrayFromImage(HA100_mask)

    HU_800 = np.average(HA800_array, weights=HA800_mask_array)
    print('HU_800: '+str(HU_800))
    HU_400 = np.average(HA400_array, weights=HA400_mask_array)
    print('HU_400: '+str(HU_400))
    HU_100 = np.average(HA100_array, weights=HA100_mask_array)
    print('HU_100: '+str(HU_100))

    x = [HU_100, HU_400, HU_800]
    y = [100, 400, 800]
    m,b = np.polyfit(x, y, 1)
    print('m: '+str(m))
    print('b: '+str(b))

    #save the slope and offset values to csv file
    d = {'0 slope': [m], '11 offset': [b]}
    df = pd.DataFrame(data=d)
    df.to_csv('CalibrationSlope&Offset_StandardSEQCT.csv',mode='a')

    #Convert image from HU to mgHA using best fit line from phantom.
    mono40_array = sitk.GetArrayFromImage(mono_40)
    mgHA_array = m*mono40_array + b
    img_HA = sitk.GetImageFromArray(mgHA_array)
    img_HA.SetSpacing(img_spacing)
    img_HA.SetOrigin(img_origin)


    sitk.WriteImage(img_HA,filePath+'/'+'Calibrated_StandardSEQCT.mha',True)


def main():
    # Set up description
    description='''Function to calibrate bone density images using standard single-energy QCT (SEQCT) method.

    This program will read in GSI image (generated by GE = simulation of standard SEQCT at 120 keV), as well as a mask image
    indicating the location of calibration rods (mgHA). This script is meant to be used on images scanned with
    QRM calibration phantom. Phantom mask image should have the following values:
    800 mgHA rod label = 3
    400 mgHA rod label = 2
    100 mgHA rod label = 1

    The output will be an image that's calibrated using a standard calibration method, such that hounsfield units are converted to mgHA/ccm based on
    correlation to the QRM phantom.

'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="SEQCT_PhantomCalibration",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("--GSI_filename","-le",
                        default='GSI',
                        type=str,
                        help="Filename for the GSI image")
    parser.add_argument("--HA_mask_fnm","-m",
                        default='HA_mask',
                        type=str,
                        help="Filename for the HA rod mask image")


    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('SEQCT_PhantomCalibration', vars(args)))

    # Run program
    SEQCT_Calibration(**vars(args))


if __name__ == '__main__':
    main()