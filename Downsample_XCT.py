import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from bonelab.util.echo_arguments import echo_arguments


def downsample_img(filePath,img_fnm,downsample_factor):

    xct = sitk.ReadImage(filePath+'/'+img_fnm+'.mha', sitk.sitkFloat32)

    xct_spacing = xct.GetSpacing()
    new_spacing = [xct_spacing[0]*downsample_factor, xct_spacing[1]*downsample_factor, xct_spacing[2]*downsample_factor]
    xct_size = xct.GetSize()
    new_size = [int(xct_size[0]/downsample_factor), int(xct_size[1]/downsample_factor), int(xct_size[2]/downsample_factor)]

    #resample images to isotropic voxel size for FEA:
    resample_xct = sitk.ResampleImageFilter()
    resample_xct.SetReferenceImage(xct)
    resample_xct.SetOutputSpacing(new_spacing)
    resample_xct.SetSize(new_size)
    resample_xct.SetInterpolator(sitk.sitkLinear)
    xct_s4 = resample_xct.Execute(xct)
    sitk.WriteImage(xct_s4, filePath+'/'+img_fnm+'_s'+str(downsample_factor)+'.mha',True)

def main():
    # Set up description
    description='''Function to register HR-pQCT with standard clinical CT (DECT).

    This program will read in a bilateral knee CT (acquired using SALTACII protocol), and crop the image to isolate the injured knee (identified using SegLabels csv)
    It also reads in an HR-pQCT image of the injured knee only (downsampled by a factor of 4 using the Downsample_XCT.py script), and performs a registration to align the two image sets.
    Registration requires landmark initialization: The program will require the user to manually select 4 landmarks in the fixed & moving images. (Please see DECT Analysis Manual document for details on how to select the landmarks)
    Separate registrations should be performed for the femur & tibia.


    Input images = simulated monoenergetic image (e.g., at 40 keV), downsampled HR-pQCT, mask of major bones (generated through method of Krčah, Marcel, Gábor Székely, and Rémi Blanc, IEEE, 2011.)


'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Downsample_XCT",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("--img_fnm","-if",
                        default='XCT_img',
                        type=str,
                        help="Filename for the HR-pQCT image")
    parser.add_argument("--downsample_factor","-df",
                        default=4,
                        type=int,
                        help="By what factor to downsample")

    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Downsample_XCT', vars(args)))

    # Run program
    downsample_img(**vars(args))


if __name__ == '__main__':
    main()
