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

plt.imshow(anat[:, 80, :])
plt.imshow(func[:, 50, :])



## hough transform columwise

import numpy as np
import cv2

img = func[:, 120, :]/256
img[np.where(img<1)]=0
img_bw = img.copy()
img_bw[np.where(img>1)]=255
img = np.uint8(img)
img_bw = np.uint8(img_bw)
cv2.imshow('Original',img)

edges = cv2.Canny(img_bw,250,500,apertureSize = 3)
cv2.imshow('Edges',edges)
lines = cv2.HoughLines(edges,1,np.pi/180,70)

img_rslt = np.zeros([img.shape[0],img.shape[1], 3])
img_rslt[:,:,0] = img; img_rslt[:,:,1] = img; img_rslt[:,:,2] = img;
img_rslt = np.uint8(img_rslt)
cv2.imshow('img_rslt',img_rslt)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #img = np.float32(img)
    cv2.line(img_rslt,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('Lines',img_rslt)
theta_c = theta.copy()


## obtain rotation matrix, columwise
import math 
rot_deg_column = math.degrees(theta_c) + 90
# grab the dimensions of the image and calculate the center of the image
tmpimg = func[:, 0, :]
(h_col, w_col) = tmpimg.shape[:2]
(cX_col, cY_col) = (w_col / 2, h_col / 2)
 
# rotate our image by 45 degrees
M_column = cv2.getRotationMatrix2D((cX_col, cY_col), rot_deg_column, 1.0)
#rotated = cv2.warpAffine(img, M, (w, h))
#cv2.imshow("Rotated by 45 Degrees", rotated)

## volume rotation, column wise
func_rot_result = np.zeros(func.shape)
for c_index in range(0,func.shape[1]):
   tmpimg = func[:, c_index, :]/256
   tmpimg = np.uint8(tmpimg)
   func_rot_result[:,c_index,:] = cv2.warpAffine(tmpimg, M_column, (w_col, h_col))
   
      
plt.imshow(func_rot_result[:,100,:])   
   
   

   
   
## hough transform framewise
import numpy as np
import cv2

img = func_rot_result[:, :, 120]
img[np.where(img<1)]=0
img_bw = img.copy()
img_bw[np.where(img>1)]=255
img = np.uint8(img)
img_bw = np.uint8(img_bw)
cv2.imshow('Original',img)

edges = cv2.Canny(img_bw,250,500,apertureSize = 3)
cv2.imshow('Edges',edges)
lines = cv2.HoughLines(edges,1,np.pi/180,70)

img_rslt = np.zeros([img.shape[0],img.shape[1], 3])
img_rslt[:,:,0] = img; img_rslt[:,:,1] = img; img_rslt[:,:,2] = img;
img_rslt = np.uint8(img_rslt)
cv2.imshow('img_rslt',img_rslt)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #img = np.float32(img)
    cv2.line(img_rslt,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('Lines',img_rslt)
theta_f = theta.copy()

## obtain rotation matrix
import math 
rot_deg_frame = math.degrees(theta_f) + 270
# grab the dimensions of the image and calculate the center of the image
tmpimg = func_rot_result[:, :, 0]
(h_frame, w_frame) = tmpimg.shape[:2]
(cX_frame, cY_frame) = (w_frame / 2, h_frame / 2)
 
# rotate our image by 45 degrees
M_frame = cv2.getRotationMatrix2D((cX_frame, cY_frame), rot_deg_frame, 1.0)
#rotated = cv2.warpAffine(img, M, (w, h))
#cv2.imshow("Rotated by 45 Degrees", rotated)
      
## rotation, row wise

for f_index in range(0,func.shape[2]):  
   tmpimg = func_rot_result[:, :, f_index]
   tmpimg = np.uint8(tmpimg)
   func_rot_result[:,:,f_index] = cv2.warpAffine(tmpimg, M_frame, (w_frame, h_frame))

   

























