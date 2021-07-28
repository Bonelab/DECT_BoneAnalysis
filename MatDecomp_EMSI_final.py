# Imports:
import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from bonelab.util.echo_arguments import echo_arguments
# import Mono_ScatterPlotAnalysis_v1 as spa
# import MatDecomp_library_v1 as mdc

def MatDecomp_EMSI(filePath,lowenergy_filename,highenergy_filename,marrow_attenuation_low,marrow_attenuation_high,edema_attenuation_low,edema_attenuation_high,bone_attenuation_low,bone_attenuation_high):

    img_low = sitk.ReadImage(filePath+'/'+lowenergy_filename+'.mha', sitk.sitkFloat32)
    img_high = sitk.ReadImage(filePath+'/'+highenergy_filename+'.mha', sitk.sitkFloat32)

    low_energy = sitk.GetArrayFromImage(img_low)
    high_energy = sitk.GetArrayFromImage(img_high)
    img_size = img_low.GetSize()
    img_spacing = img_low.GetSpacing()
    img_origin = img_low.GetOrigin()

    a= marrow_attenuation_low #fat: low energy
    b= edema_attenuation_low #muscle: low energy
    c= bone_attenuation_low #HAP: low energy
    d=low_energy  #low energy image array
    e= marrow_attenuation_high #fat: high energy
    f= edema_attenuation_high #muscle: high energy
    g= bone_attenuation_high #HAP: high energy
    h=high_energy  #high energy image array

    #This part modified from Doug Kondro's code:
    xsize = img_size[2]
    ysize = img_size[1]
    zsize = img_size[0]

    #solve system of equations for 3 mat decomp
    Z=np.zeros((xsize, ysize, zsize,3))
    Z[:,:,:,1]=((d-a)-(h-e)*(c-a)/(g-e))/((b-a)-(c-a)*(f-e)/(g-e))
    Z[:,:,:,2]=((d-a)-(h-e)*(b-a)/(f-e))/((c-a)-(b-a)*(g-e)/(f-e))
    Z[:,:,:,0]=1-Z[:,:,:,2]-Z[:,:,:,1]

    #remove air:
    Z=Z.astype(float)
    low_energy=low_energy.astype(float)
    Z[low_energy<-300.00,:]=0


    #remove double negatives:
    print('removing double negatives')
    Z_DN=((Z[:,:,:,0]*Z[:,:,:,1]*Z[:,:,:,2])>0) & ((np.absolute(Z[:,:,:,0])+np.absolute(Z[:,:,:,1])+np.absolute(Z[:,:,:,2]))>1)
    Z_S=((Z[:,:,:,0]*Z[:,:,:,1]*Z[:,:,:,2])>0) & ((np.absolute(Z[:,:,:,0])+np.absolute(Z[:,:,:,1])+np.absolute(Z[:,:,:,2]))==1)


    print(np.sum(Z_DN))
    #Everything Negative is set to zero
    #And set the remaining positive to 1 if it had a double negative
    Z[(Z_DN) & (Z[:,:,:,0]>1.00),0] =1.00
    Z[(Z_DN) & (Z[:,:,:,1]>1.00),1] =1.00
    Z[(Z_DN) & (Z[:,:,:,2]>1.00),2] =1.00
    Z[Z<0]=0


    #Identify the points that need to be interpolated, the ones with two numbers
    Z_fat=((Z[:,:,:,1]+Z[:,:,:,2])>1.00) & (Z[:,:,:,1]*Z[:,:,:,2]>0)
    Z_muscle=((Z[:,:,:,0]+Z[:,:,:,2])>1.00) & (Z[:,:,:,0]*Z[:,:,:,2]>0)
    Z_HA=((Z[:,:,:,0]+Z[:,:,:,1])>1.00) & (Z[:,:,:,0]*Z[:,:,:,1]>0)

    #print("Finding the zero fat's")
    Z[Z_fat,1]=project_percent_a_array(low_energy[Z_fat],high_energy[Z_fat],edema_attenuation_low,edema_attenuation_high,bone_attenuation_low,bone_attenuation_high)
    Z[Z_fat,2]=1-Z[Z_fat,1]

    #print("Finding the zero muscle's")
    Z[Z_muscle,0]=project_percent_a_array(low_energy[Z_muscle],high_energy[Z_muscle],marrow_attenuation_low,marrow_attenuation_high,bone_attenuation_low,bone_attenuation_high)
    Z[Z_muscle,2]=1-Z[Z_muscle,0]

    #print("Finding the zero HAs")
    Z[Z_HA,0]=project_percent_a_array(low_energy[Z_HA],high_energy[Z_HA],marrow_attenuation_low,marrow_attenuation_high,edema_attenuation_low,edema_attenuation_high)
    Z[Z_HA,1]=1-Z[Z_HA,0]

    img_fat2 = sitk.GetImageFromArray(Z[:,:,:,0]*1000)
    img_muscle2 = sitk.GetImageFromArray(Z[:,:,:,1]*1000)
    img_HAP2 = sitk.GetImageFromArray(Z[:,:,:,2]*1000)

    img_fat2.SetSpacing(img_spacing)
    img_muscle2.SetSpacing(img_spacing)
    img_HAP2.SetSpacing(img_spacing)

    img_fat2.SetOrigin(img_origin)
    img_muscle2.SetOrigin(img_origin)
    img_HAP2.SetOrigin(img_origin)

    sitk.WriteImage(img_fat2,filePath+'/'+'marrow_fraction.mha',True)
    sitk.WriteImage(img_muscle2,filePath+'/'+'edema_fraction.mha',True)
    sitk.WriteImage(img_HAP2,filePath+'/'+'HA_fraction.mha',True)


def project_percent_a_array(point_x,point_y,a_x,a_y,b_x,b_y):
    #Modified from Doug Kondro's code

    #We find the line between the two input points
    m=(b_y-a_y)/(b_x-a_x)
    b=b_y-m*b_x
    #Our projection must be an orthogonal line to this
    m_orthogonal=-1/m
    b_orthogonal=point_y-m_orthogonal*point_x
    #Find where they intercept
    x_intercept=(b_orthogonal-b)/(m-m_orthogonal)
    y_intercept=m_orthogonal*x_intercept+b_orthogonal
    #Find the distances
    a_2_b=np.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
    b_2_intercept=np.sqrt((b_x-x_intercept)**2+(b_y-y_intercept)**2)
    a_2_intercept=np.sqrt((a_x-x_intercept)**2+(a_y-y_intercept)**2)
    percent_a=b_2_intercept/a_2_b
    #If our x point is to the right point b and the slope is positive and point a is the lower point set to 1
    #percent_a[((b_x-a_x)*b*(x_intercept-a_x))<0]=1
    percent_a[b_2_intercept>a_2_b]=1 #Only works if the distance is not outside of length from a to b
    #print(np.sum(a_2_intercept>a_2_b))
    percent_a[a_2_intercept>a_2_b]=0
    if (np.sum(percent_a > 1))>0:
        print("There are some precentages greater than 1")

    return percent_a

def extend_x_y(a_x,a_y,b_x,b_y,extension):
    #point x and point y are arrays
    c_x=(b_x-a_x)*(extension+1)+a_x
    c_y=(b_y-a_y)*(extension+1)+a_y

    return c_x,c_y


def main():
    # Set up description
    description='''DE-CT based 3-material decomposition for bone marrow edema imaging

This program will read in 2 simulated monochromatic images (generated by GE),
and will perform material decomposition to generate 3 images corresponding to
the fraction of each voxel consisting of bone, normal marrow, and edema.

Default values for  monochromatic energies and material decomposition parameters
are those optimized from the KneeBML study. However, different parameters can be
provided.

This method has been published as de Bakker et al. (2021) "A quantitative assessment of dual energy computed tomography-based material decomposition for imaging bone marrow edema associated with acute knee injury" Med Phys
DOI: https://doi.org/10.1002/mp.14791



'''

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="blMatDecomp_EMSI",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("--lowenergy_filename","-le",
                        default='40keV',
                        type=str,
                        help="Filename for the low energy simulated monochromatic image")
    parser.add_argument("--highenergy_filename","-he",
                        default='90keV',
                        type=str,
                        help="Filename for the high energy simulated monochromatic image")
    parser.add_argument("--marrow_attenuation_low","-ml",
                        default=-149.0,
                        type=float,
                        help="attenuation of marrow tissue at low energy (in HU)")
    parser.add_argument("--marrow_attenuation_high","-mh",
                        default=-100.0,
                        type=float,
                        help="attenuation of marrow tissue at high energy (in HU)")
    parser.add_argument("--edema_attenuation_low","-el",
                        default=85.0,
                        type=float,
                        help="attenuation of edema at low energy (in HU)")
    parser.add_argument("--edema_attenuation_high","-eh",
                        default=36.0,
                        type=float,
                        help="attenuation of edema at high energy (in HU)")
    parser.add_argument("--bone_attenuation_low","-bl",
                        default=3360.0,
                        type=float,
                        help="attenuation of HA phantom at low energy (in HU)")
    parser.add_argument("--bone_attenuation_high","-bh",
                        default=980.0,
                        type=float,
                        help="attenuation of HA phantom at high energy (in HU)")




    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('MatDecomp_EMSI', vars(args)))

    # Run program
    MatDecomp_EMSI(**vars(args))

    print("Please cite 'de Bakker et al. Med Phys 2021' when using this analysis.")
    print("https://pubmed.ncbi.nlm.nih.gov/33606278/")

if __name__ == '__main__':
    main()
