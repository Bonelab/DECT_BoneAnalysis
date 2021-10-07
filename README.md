### Functions using dual-energy CT (DECT) for bone imaging:

Two major functions:
1.) Optimized 3-material decomposition for edema imaging and edema segmentation
2.) DECT based bone density calibration

All functions are based on simulated monoenergetic images at 2 energy levels (e.g., 40 and 90 keV)

Also includes functions for image registration:
1.) MRI to DECT
2.) HR-pQCT to DECT
3.) Contralateral knee to injured knee

##### For in-depth instructions:
Bone Lab members: Please see BOYD drive: /BOYD/04 - Resources/05 - Tutorials & Training/DECT
Others: Please contact the [Bone Imaging Lab](https://www.ucalgary.ca/labs/bonelab/contact)



To run 3-material decomposition:
MatDecomp_EMSI_final.py -- runs basic material decomposition, optimized for edema imaging
Segment_Edema -- applies non-local means filter to edema image, and applies bone marrow mask and threshold to segment edema region

For DECT-based bone density calibration:
Scan using density calibration phantom, create mask for density rods
Then run DECT_PhantomCalibration
