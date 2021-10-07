### Functions using dual-energy CT (DECT) for bone imaging:

Two major functions:
1. Optimized 3-material decomposition for edema imaging and edema segmentation
2. DECT based bone density calibration

All functions are based on simulated monoenergetic images at 2 energy levels (e.g., 40 and 90 keV)

Also includes functions for image registration:
1. MRI to DECT
2. HR-pQCT to DECT
3. Contralateral knee to injured knee

##### For in-depth instructions:
- Bone Lab members: Please see BOYD drive: /BOYD/04 - Resources/05 - Tutorials & Training/DECT
- Others: Please contact the [Bone Imaging Lab](https://www.ucalgary.ca/labs/bonelab/contact)


#### Quick Start

##### To run 3-material decomposition:
1. Generate simulated monoenergetic images at 40 and 90 keV
2. Convert DICOMs to MHA: dicom_Converter_batch.py
3. Run basic material decomposition, optimized for edema imaging: MatDecomp_EMSI_final.py
4. Segment marrow space:
   1. Generate periosteal segmentation (e.g., using FemurSegmentation repository)
   2. Calibrate CT images: Standard_SEQCT_PhantomCalibration_GSI.py
   3. Run EndostealSegmentation_LANG.py
5. Filter  edema image, then apply bone marrow mask and threshold to segment edema region: Segment_Edema.py

##### For DECT-based bone density calibration:
- Scan using density calibration phantom
- Create mask for density rods
- Run DECT_PhantomCalibration.py

##### For image registrations (Note: Tibia and Femur should always be registered separately):
- MRI to DECT registrations:
  - Flip sagittal MRI into axial coordinate system to match DECT: Flip_Coordinates_SagittalToAxial.py
  - Run the appropriate registration script:
    - To register PD or T1 MRI to DECT: Register_PDMRI-DECT_LandmarkInitialized.py
    - To register T2 MRI to DECT:
      1. First register PD/T1 to DECT
      2. Registser T2 to PD/T1: Register_T2-PDMRI.py
      3. Combine results of steps 1 and 2 above: Transform_T2MRI-DECT.py
- HR-pQCT to DECT registrations:
  1. Convert HR-pQCT aims to mha: Aim2Mha_py3.py
  2. Downsample HR-pQCT images by a factor of 4: Downsample_XCT.py
  3. Run registration: Register_XCT-DECT_LandmarkInitialized.py
- Contralateral registrations: run Register_Injured-Contralateral.py
