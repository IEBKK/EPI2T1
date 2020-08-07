#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:24:20 2020

@author: mingeon kim
"""


import numpy as np # linear algebra
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
from nilearn.image import load_img

#mask_filename = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
subject_filename = '/home/bk/Desktop/EPI2T1/filtered2hires_exam.nii.gz'
#spm_filename = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
smri_filename = '/home/bk/Desktop/EPI2T1/primeRecon_02_t1_mprage_brain.nii.gz'

smri_img = nl.image.load_img(smri_filename)
func_img = nl.image.load_img(subject_filename)

anat = smri_img.get_fdata()
func = func_img.get_fdata()

plt.imshow(anat[80, :, :])
plt.imshow(func[80, :, :])