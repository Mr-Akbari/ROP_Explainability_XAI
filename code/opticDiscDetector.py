#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:35:26 2018

@author: veysiyildiz

detect the optic disc center of the input image 
"""
import os
from keras.models import model_from_json
import loadImages
from configs.utils.config_utils import process_config
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from configs.utils.img_utils import get_test_patches,pred_to_patches,recompone_overlap,imgResize,postprocess
from configs.utils.utils import visualize,gray2binary
def analyze_name(path):
	return (path.split('\\')[-1])
        
def find_optic_disc_center(image_name,image_folder):
    print('[INFO] Reading Configs...')
    config = None
    try:
        config = process_config('code/configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
        
    model = model_from_json(open('parameters/unet_opticDisc/optic_disc_detector_model.json').read())
    model.load_weights('parameters/unet_opticDisc/optic_disc_detector_model.h5')
    binaryResult,probResult = [], []

    path = image_folder + image_name
    orgImg_temp=plt.imread(path)
    if orgImg_temp.shape[0]>1000 or orgImg_temp.shape[1]>1000:
        orgImg_temp=imgResize(orgImg_temp,0.5)
    orgImg=orgImg_temp[:,:,1]*0.75+orgImg_temp[:,:,0]*0.25
    print("[Info] Analyze filename...",analyze_name(path))
    height,width=orgImg.shape[:2]
    orgImg = np.reshape(orgImg, (height,width,1))
    patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,config)

    predictions = model.predict(patches_pred, batch_size=32, verbose=1)
    pred_patches=pred_to_patches(predictions,config)

    pred_imgs=recompone_overlap(pred_patches,config,new_height,new_width)
    pred_imgs=pred_imgs[:,0:height,0:width,:]


    probResult=pred_imgs[0,:,:,0]
    binaryResult=gray2binary(probResult,threshold=0.5)

    binaryResult,probResult=postprocess(binaryResult,probResult) # post process to get the optic disc
    
    
    contours, hierarchy  = cv2.findContours(binaryResult, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours))
    if len(contours) == 0:
        print('Disc center Not found!')
        return
    image_temp = binaryResult.copy()
    M = cv2.moments(contours[0])
    cx = cy = 0
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00']) 
        cv2.circle(image_temp, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(image_temp, "center", (cx - 20, cy - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imwrite(image_folder+image_name.split('.')[0]+"_center.jpg", image_temp*255)
        adjustImg = cv2.drawContours(orgImg_temp, contours, -1, (0, 255, 0), 2)
        resultMerge=visualize([adjustImg/255.,binaryResult],[1,2])
        resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_folder+image_name.split('.')[0]+"_merge.jpg",resultMerge)
        cv2.imwrite(image_folder+image_name.split('.')[0]+ "_prob.bmp", (probResult*255).astype(np.uint8))
    return np.array([cx,cy])