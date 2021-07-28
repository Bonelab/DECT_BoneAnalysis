#### Functions using dual-energy CT (DECT):

##Two major functions:
1.) Optimized 3-material decomposition for edema imaging
2.) DECT based bone density calibration

All functions are based on simulated monoenergetic images at 2 energy levels (e.g., 40 and 90 keV)

To run 3-material decomposition:
MatDecomp_EMSI_final.py -- runs basic material decomposition, optimized for edema imaging
Segment_Edema -- applies non-local means filter to edema image, and applies bone marrow mask and threshold to segment edema region

For DECT-based bone density calibration:
Scan using density calibration phantom, create mask for density rods
Then run DECT_PhantomCalibration
