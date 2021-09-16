import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def FindLabels(patient_id):
    label_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4))

    if patient_id == "SALTACII_0037":
        patient_num = 36
    elif patient_id == "SALTACII_0040":
        patient_num = 37
    else:
        patient_num = int(patient_id[len(patient_id)-4:])

    f_i = label_mat[patient_num-1,0]
    t_i = label_mat[patient_num-1,1]
    f_c = label_mat[patient_num-1,2]
    t_c = label_mat[patient_num-1,3]

    return f_i,t_i,f_c,t_c

def FindInjuredSide(patient_id):
    side_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(4,5))

    if patient_id == "SALTACII_0037":
        patient_num = 36
    elif patient_id == "SALTACII_0040":
        patient_num = 37
    else:
        patient_num = int(patient_id[len(patient_id)-4:])

    side_num = side_mat[patient_num-1,1]
    return side_num

def FindMRIThreshold(patient_id):
    PeakIntensity_mat = np.loadtxt(open("SALTACII_SegLabels.csv", "rb"), delimiter=",", skiprows=1, usecols=(5,6))

    if patient_id == "SALTACII_0037":
        patient_num = 36
    elif patient_id == "SALTACII_0040":
        patient_num = 37
    else:
        patient_num = int(patient_id[len(patient_id)-4:])

    PeakIntensity = PeakIntensity_mat[patient_num-1,1]
    MRI_thres = PeakIntensity*0.2
    # print(MRI_thres)
    return MRI_thres
