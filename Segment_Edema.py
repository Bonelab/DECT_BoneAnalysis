import SimpleITK as sitk
import sys
import os
import numpy as np
import argparse
from bonelab.util.echo_arguments import echo_arguments



def NLM_filter(filePath,CT_SoftTissue_fraction):
    edema_array = sitk.GetArrayFromImage(CT_SoftTissue_fraction)


    img_spacing = CT_SoftTissue_fraction.GetSpacing()
    img_origin = CT_SoftTissue_fraction.GetOrigin()
    img_size = CT_SoftTissue_fraction.GetSize()
    print(img_size)

    #Sigma values:
    sigma_edema = 30
    sigma_fat = 30
    sigma_bone = 30


    Wp = 1 #for window size 3x3
    Ws = 5 #for window size 11x11
    E = 3 #using 3 energies
    h = 5

    exp_wt = -1.0/(E*(Wp*2+1)**3*h**2)

    # print(exp_wt)
    Wp_length = (Wp*2+1)**3
    # print(Wp_length)

    Z=np.zeros((Ws*2+1, Ws*2+1, Ws*2+1,3))
    P=np.zeros((Wp*2+1, Wp*2+1, Wp*2+1,3))
    edema_array_filtered_edemaonly = np.zeros((img_size[2],img_size[1],img_size[0]))
    edema_patch_array = np.zeros((img_size[2],img_size[1],img_size[0],Wp_length))


    for i in range(Ws+Wp,img_size[2]-Ws-Wp-1):
        for j in range(Ws+Wp,img_size[1]-Ws-Wp-1):
            for k in range(Ws+Wp,img_size[1]-Ws-Wp-1):

                edema_patch = edema_array[i-Wp:i+Wp+1,j-Wp:j+Wp+1,k-Wp:k+Wp+1]
                edema_patch_array[i,j,k,:] = np.reshape(edema_patch,[1,Wp_length])



    for i in range(Ws+Wp,img_size[2]-Ws-Wp-1):
        for j in range(Ws+Wp,img_size[1]-Ws-Wp-1):
            for k in range(Ws+Wp,img_size[1]-Ws-Wp-1):

                edema_search = edema_array[i-Ws:i+Ws+1,j-Ws:j+Ws+1,k-Ws:k+Ws+1]


                voxel_location = [i,j,k]
                sigmas = [sigma_edema,sigma_fat,sigma_bone]

                neighbor_weights_edemaonly = np.sum((edema_patch_array[i-Ws:i+Ws+1,j-Ws:j+Ws+1,k-Ws:k+Ws+1,:]-edema_patch_array[i,j,k,:])**2/sigma_edema**2,axis=3)
                edema_array_filtered_edemaonly[i,j,k] = np.sum(edema_search*np.exp(exp_wt*3*neighbor_weights_edemaonly))/np.sum(np.exp(exp_wt*3*neighbor_weights_edemaonly))


    edema_img_filtered_edemaonly = sitk.GetImageFromArray(edema_array_filtered_edemaonly)
    edema_img_filtered_edemaonly.SetSpacing(img_spacing)
    edema_img_filtered_edemaonly.SetOrigin(img_origin)
    sitk.WriteImage(edema_img_filtered_edemaonly,filePath+'/'+'edema_NLMFiltered.mha', True)

    return edema_img_filtered_edemaonly


def threshold_edema_3mat(filePath,edema_fraction_fnm, mask_fnm,bone_id,injured_side,use_contralateral):
    #define thresholds
    #these thresholds determined based on optimizing the method for comparison of DECT to MRI using the first 10 SALTACII scans
    if bone_id == 'Femur':
        lower_thres = 735
        bone_label = 1
    elif bone_id == 'Tibia':
        lower_thres = 616
        bone_label = 2

    #read in the edema fraction image
    muscle_fraction = sitk.ReadImage(filePath+'/'+edema_fraction_fnm+'.mha', sitk.sitkFloat32)

    #apply non-local means filter to smooth edema fraction image:
    muscle_fraction = NLM_filter(filePath,muscle_fraction)

    #Simple threshold of edema fraction image: >= lower_thres = edema:
    thres_edema = muscle_fraction > lower_thres
    sitk.WriteImage(thres_edema,filePath+'/'+'edema_thres_raw.mha',True)

    #mask by endosteal mask for the bone of interest:
    marrow_mask = sitk.ReadImage(filePath+'/'+mask_fnm+'.mha', sitk.sitkFloat32)
    marrow_mask_bin = marrow_mask==bone_label

    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius(3)
    erode.SetForegroundValue(1)
    marrow_mask_er = erode.Execute(marrow_mask_bin)

    sitk.WriteImage(marrow_mask_er,'marrow_mask_er.mha')

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)
    edema_masked = marrow_mask_er*thres_edema


    #Component labeling to remove small clusters less than 100 voxels in size:
    cc_filter = sitk.ConnectedComponentImageFilter()
    cl = cc_filter.Execute(edema_masked)

    cc_relabel = sitk.RelabelComponentImageFilter()
    cc_relabel.SetMinimumObjectSize(100)
    cl_relabeled = cc_relabel.Execute(cl)
    cl_relabeled_thres = cl_relabeled >=1

    sitk.WriteImage(cl_relabeled_thres, filePath+'/'+'edema_thres_final_'+bone_id+'.mha',True)

def TransformContra(img_injured,bone_id,filePath):
    #Crop flipped filtered edema image and transform to align contralateral bone with injured side:
    flipped_fnm = 'edema_NLMFiltered_flipped'
    # flipped_fnm = 'edema_NLMFiltered_flipped'
    # original_fnm = 'edema_NLMFiltered'
    img_flipped = sitk.ReadImage(filePath+'/'+flipped_fnm+'.mha',sitk.sitkFloat32)
    # img_org = sitk.ReadImage(filePath+'/'+original_fnm+'.mha',sitk.sitkFloat32)

    crop_DE = sitk.CropImageFilter()

    flipped_size = img_flipped.GetSize()
    if injured_side == 'left':
        crop_DE.SetLowerBoundaryCropSize([np.int(flipped_size[0]/2),0,0])
        crop_DE.SetUpperBoundaryCropSize([0,0,0])
    elif injured_side == 'right':
        crop_DE.SetLowerBoundaryCropSize([0,0,0])
        crop_DE.SetUpperBoundaryCropSize([np.int(flipped_size[0]/2),0,0])

    img_flipped = crop_DE.Execute(img_flipped)
    img_flipped. SetOrigin([0,0,0])


    tfm_name = 'rigid_registration_contra_'+bone_id
    contra_tfm = sitk.ReadTransform(filePath+'/'+bone_id+'/'+tfm_name+'.tfm')

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img_flipped)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(contra_tfm)
    contra_registered = resample.Execute(img_flipped)

    sitk.WriteImage(contra_registered,filePath+'/'+bone_id+'/'+'/Edema_Contra_Registered.mha',True)

    return contra_registered

def threshold_edema_subtractcontra(filePath,edema_fraction_fnm,mask_fnm,bone_id,injured_side,use_contralateral):
    #these thresholds determined based on optimizing the subtract_contra method for comparison of DECT to MRI using the first 10 SALTACII scans
    if bone_id == 'Femur':
        lower_thres = 682
    elif bone_id == 'Tibia':
        lower_thres = 562

    #read in the edema fraction image
    muscle_fraction = sitk.ReadImage(filePath+'/'+edema_fraction_fnm+'.mha', sitk.sitkFloat32)

    #apply non-local means filter to smooth edema fraction image:
    muscle_fraction = NLM_filter(filePath,muscle_fraction)


    #mask by endosteal mask for the bone of interest:
    marrow_mask = sitk.ReadImage(filePath+'/'+mask_fnm+'.mha', sitk.sitkFloat32)
    if bone_id == "Femur":
        marrow_mask_bin = marrow_mask==1
    elif bone_id == "Tibia":
        marrow_mask_bin = marrow_mask==2

    contra_img = TransformContra(muscle_fraction,participant_id,bone_id,filePath)

    #filter contra image to remove some noise:
    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetMaximumKernelWidth(2)
    gauss_filter.SetVariance(2)

    contra_img = gauss_filter.Execute(contra_img)

    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkFloat32)

    injured_masked_img = injured_img*cast.Execute(endosteal_mask)
    contra_masked_img = contra_img*cast.Execute(endosteal_mask)

    subtract_thres = 150 #edema fraction of 15% over contralateral appears to work best (compared to 10% and 20%)
    elevatedsignal_mask = injured_masked_img > lower_thres

    subtract_contra = injured_masked_img - contra_masked_img

    img_elevated = injured_masked_img * cast.Execute(elevatedsignal_mask)
    sub_img_elevated = subtract_contra * cast.Execute(elevatedsignal_mask)

    thres_img = sub_img_elevated > subtract_thres

    sitk.WriteImage(elevatedsignal_mask, filePath+'/'+bone_id+'/elevated_signal_mask_boundMRI.mha',True)
    sitk.WriteImage(thres_img,filePath+'/'+bone_id+'/double_thres_img_boundMRI.mha',True)

    cc_filter = sitk.ConnectedComponentImageFilter()
    cl = cc_filter.Execute(thres_img)
    # cl = cc_filter.Execute(edema_dil)

    cc_relabel = sitk.RelabelComponentImageFilter()
    # cc_relabel.SetMinimumObjectSize(250)
    cc_relabel.SetMinimumObjectSize(100)
    cl_relabeled = cc_relabel.Execute(cl)
    cl_relabeled_thres = cl_relabeled >=1

    CT_edema_image = cl_relabeled_thres
    sitk.WriteImage(CT_edema_image, filePath+'/'+bone_id+'/'+participant_id+'_edema_NLMFiltered_cc2_doublethres.mha',True)


def main():
    # Set up description
    description='''Function to segment edema based on 3-material decomposition.

    This program will read in the edema fraction image from 3-material decomposition, as well
    as the endosteal mask for the bone of interest, and will segment out edema by applying a non-local means filter, then applying a threshold,
    masking out the non-marrow regions of the image, and performing a component labeling step to remove noise.


    Input image = edema fraction from 3-material decompsition (for best results, use the included non-local means filter function to smooth the edema fraction image)

    Default threshold value is the optimized threshold value from the SALTACII study. However, this can
    be adjusted for other applications.

    Also includes optional functions (threshold_edema_subtractcontra and TransformContra), which can be applied in bilateral scans to improve
    sensitivity of edema detection. These functions will threshold the edema fraction image after subtracting out the contralateral knee
    (allows to account for normal variations in bone marrow fluid content -- improves imaging sensitivity and reduces noise in segmented edema image)
    This function still needs to be fully incorporated in this script. Prior to applying this step, must first register the contralateral knee to align with the injured knee (i.e., using Register_Injured-Contralateral.py).

    This method has been published as de Bakker et al. (2021) "A quantitative assessment of dual energy computed tomography-based material decomposition for imaging bone marrow edema associated with acute knee injury" Med Phys
    DOI: https://doi.org/10.1002/mp.14791

'''


    # Set up argument parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="Segment_Edema",
        description=description
    )

    parser.add_argument("filePath",
                        type=str,
                        help="The filepath")
    parser.add_argument("--edema_fraction_fnm","-ef",
                        default='edema_fraction',
                        type=str,
                        help="Filename for the edema fraction image")
    parser.add_argument("--mask_fnm","-m",
                        default='Calibrated_StandardSEQCT_LANG',
                        type=str,
                        help="Filename for the marrow mask image")
    parser.add_argument("--bone_id","-b",
                        default='Tibia',
                        type=str,
                        help="Bone of interest (Femur or Tibia)")
    parser.add_argument("--use_contralateral","-uc",
                        default='N',
                        type=str,
                        help="Use data from the contralateral knee to improve the segmentation (Y or N)")
    parser.add_argument("--injured_side","-i",
                        default='left',
                        type=str,
                        help="Side of injured knee (left or right)")


    # Parse and display
    args = parser.parse_args()
    print(echo_arguments('Segment_Edema', vars(args)))

    # Run program
    if args.use_contralateral == "N":
        threshold_edema_3mat(**vars(args))
    elif args.use_contralateral == "Y":
        threshold_edema_subtractcontra(**vars(args))


if __name__ == '__main__':
    main()
