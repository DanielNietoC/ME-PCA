#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23 2024

@author: mflores
"""

from nilearn import image
from nilearn import signal
from nilearn.masking import apply_mask
from nilearn.maskers import NiftiMasker
from math import prod
from scipy import stats
import numpy as np
import pandas as pd
import os
import argparse

###### Arguments ########################################################3###########
#parser=argparse.ArgumentParser(description="""Generates histograms from fmri 
#                              volumes""")
#parser.add_argument("--source_directory", default=None, type=str,
#                    help="Full path to the BIDS directory")
#
#parser.add_argument("--directories", default=None, type=list,
#                    help="""List of subdirectories inside bids directory
#                    that will be considered as different factors to plot histograms""")
#
#parser.add_argument("--oc_extention", default=None, type=str,
#                    help="""Extention of the target OC file to be look upon
#                    i.e., if my target file has the name:
#                    sub-001_task-HABLA1200_masked_epi_gm_ocDenoised.nii.gz
#                    then, OC_extention=task-HABLA1200_masked_epi_gm_ocDenoised.nii.gz""")
#
#parser.add_argument("--tsnr_extention", default=None, type=str,
#                    help="""Extention of the target tsnr file to be look upon
#                    i.e., if my target file has the name:
#                    sub-001_task-HABLA1200_masked_epi_gm_ocDenoised_tsnr.nii.gz
#                    then, OC_extention=task-HABLA1200_masked_epi_gm_ocDenoised_tsnr.nii.gz""")

#parser.add_argument("--tsnr_extention", default=None, type=str,
#                    help="""Extention of the target tsnr file to be look upon
#                    i.e., if my target file has the name:
#                    sub-001_task-HABLA1200_masked_epi_gm_ocDenoised_tsnr.nii.gz
#                    then, OC_extention=task-HABLA1200_masked_epi_gm_ocDenoised_tsnr.nii.gz""")
###### Find files ###################################################################
echoes=4
subject="sub-20"
ext=".nii.gz"
directory="/bcbl/home/public/MarcoMotion/HABLA2/"+subject+"/ses-1/func/"+subject+"_ses-1_task-SYLAB_run-1_echo-"
window=np.ones((4,4))#I am assuming this is a binary mask
#Converting ME epi data to tensor 
for echo in list(range(echoes)):
    tmp_filename=directory+str(echo+1)+"_part-mag_bold"+ext
    tmp_epiFile=image.load_img(tmp_filename)
    tmp_epiArray=tmp_epiFile.get_fdata()
    if echo==0:
        tensor_epi=tmp_epiArray[:,:,:,:,None]
    else:
        tensor_epi=np.concatenate((tensor_epi,tmp_epiArray[:,:,:,:,None]),axis=4)
# TODO: convert multidimentional array to tensor with tensorflow
tensor_dim=np.shape(tensor_epi)
tensor_voxel=tensor_dim[0:len(np.shape(window))]
n_voxels=prod(tensor_dim)
# reshape tensor to be a column vector 
reshaped_tensor=np.reshape(tensor_epi,-1)# Note that matlab uses fortran indexing, and python C indexing
# determine default index ordering (same order as input data and all of
# them with window indices combined in one)
vox_indices = list(range(len(window)))
mod_indices = list(range(len(window),len(tensor_dim)))

