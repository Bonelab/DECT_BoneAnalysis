import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse
from bonelab.util.echo_arguments import echo_arguments

def threshold_edema_2mat(filePath,participant_id,calcium_fnm,water_fnm,mask_fnm, bone_id,thres_slope,thres_offset):

    ca_img = sitk.ReadImage(filePath+'/'+participant_id+'_'+calcium_fnm+'.mha',sitk.sitkFloat32)
    water_img = sitk.ReadImage(filePath+'/'+participant_id+'_'+water_fnm+'.mha',sitk.sitkFloat32)

    #apply threshold line to segment edema based on calcium and water density images:
    m = thres_slope
    b = thres_offset


    thres_edema = (ca_img - m*water_img) < b
    # sitk.WriteImage(thres_edema,filePath+'/'+'edema_thres_2matdecomp.mha',True)

    #mask by endosteal mask (generated through method described by Lang et al.):
    marrow_mask = sitk.ReadImage(filePath+'/'+participant_id+'_LANG.mha', sitk.sitkFloat32)
    if bone_id == "Femur":
        marrow_mask_bin = marrow_mask==1
    elif bone_id == "Tibia":
        marrow_mask_bin = marrow_mask==2

    #erode marrow mask to remove potential errors near the edges of the cortex:
    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius(2)
    erode.SetForegroundValue(1)
    marrow_mask_er = erode.Execute(marrow_mask_bin)

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)
    edema_masked = marrow_mask_er*thres_edema
    sitk.WriteImage(edema_masked, filePath+'/'+participant_id+'_'+bone_id+'_'+'edema_thres_marrow_2matdecomp.mha',True)


    #Cluster analysis to remove small clusters less than 100 voxels in size:
    cc_filter = sitk.ConnectedComponentImageFilter()
    cl = cc_filter.Execute(edema_masked)

    cc_relabel = sitk.RelabelComponentImageFilter()
    cc_relabel.SetMinimumObjectSize(100)
    cl_relabeled = cc_relabel.Execute(cl)
    cl_relabeled_thres = cl_relabeled >=1
    cl_relabeled_largest = cl_relabeled == 1

    sitk.WriteImage(cl_relabeled_thres, filePath+'/'+participant_id+'_'+bone_id+'_'+'edema_2matdecomp_cc.mha',True)



def main():
    # Set up description
    description='''Function to segment edema based on 2-material decomposition.

    This program will read in the calcium and water density images (generated using CT manufacturer-provided software), as well
    as the endosteal mask for the bone of interest, and will segment out edema by plotting the voxel intensity values of the calcium and water density images, and applying a diagonal line dual threshold (identified by optimizing dice coefficient to registered fluid-sensitive MRI),
    masking out the non-marrow regions of the image, and performing a component labeling step to remove noise.


    Input images = calcim & water density images (generated using CT manufacturer-provided software)

    Default threshold value is the optimized threshold value from the KneeBML study. However, if needed, this can
    be adjusted for other applications by re-optimizing, as describedd in de Bakker et al., Med Phys 2021.

    Final output = _edema_2matdecomp_cc.mha image.

    This method corresponds to the 2-material decomposition in de Bakker et al. (2021) "A quantitative assessment of dual energy computed tomography-based material decomposition for imaging bone marrow edema associated with acute knee injury" Med Phys
    DOI: https://doi.org/10.1002/mp.14791

'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Segment_Edema_2MaterialDecomposition",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("participant_id",
                        type=str,
                        help="The participant ID")
    parser.add_argument("--calcium_fnm","-cf",
                        default='Calcium(Water)',
                        type=str,
                        help="Filename for the calcium density image")
    parser.add_argument("--water_fnm","-wf",
                        default='Water(Calcium)',
                        type=str,
                        help="Filename for the water density image")
    parser.add_argument("--mask_fnm","-m",
                        default='LANG',
                        type=str,
                        help="Filename for the marrow mask image")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")
    parser.add_argument("--thres_slope","-tm",
                        default=1.32,
                        type=float,
                        help="slope for threshold line (default = optimized value from KneeBML study)")
    parser.add_argument("--thres_offset","-tb",
                        default=-1282,
                        type=float,
                        help="y-intercept for threshold line (default = optimized value from KneeBML study)")


    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Segment_Edema_2MaterialDecomposition', vars(args)))

    # Run program
    threshold_edema_2mat(**vars(args))


if __name__ == '__main__':
    main()
