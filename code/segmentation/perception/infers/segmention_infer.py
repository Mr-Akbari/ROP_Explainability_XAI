#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:35:26 2018

@author: veysiyildiz

detect the optic disc center of the input image 
"""
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from segmentation.perception.bases.infer_base import InferBase
from segmentation.configs.utils.img_utils import get_test_patches,pred_to_patches,recompone_overlap
from segmentation.configs.utils.utils import visualize,gray2binary

from segmentation.configs.utils.config_utils import process_config


def analyze_name(path):
	return (path.split('\\')[-1])
    
def vessel_segmentation(image_name,image_folder):
    print('[INFO] Reading Configs...')
    config = None
    try:
        config = process_config('code/segmentation/configs/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)

    model = model_from_json(open('parameters/unet/unet_architecture.json').read())
    model.load_weights('parameters/unet/unet_best_weights.h5')
    
    path = image_folder + image_name
    orgImg_temp=plt.imread(path)
    orgImg=orgImg_temp[:,:,1]*0.75+orgImg_temp[:,:,0]*0.25
    print("[Info] Analyze filename...",analyze_name(path))
    height,width=orgImg.shape[:2]
    orgImg = np.reshape(orgImg, (height,width,1))
    patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,config)

    predictions = model.predict(patches_pred, batch_size=32, verbose=1)
    pred_patches=pred_to_patches(predictions,config)

    pred_imgs=recompone_overlap(pred_patches,config,new_height,new_width)
    pred_imgs=pred_imgs[:,0:height,0:width,:]

    adjustImg=adjustImg[0,0:height,0:width,:]
    #print(adjustImg.shape)
    probResult=pred_imgs[0,:,:,0]
    binaryResult=gray2binary(probResult)
    resultMerge=visualize([adjustImg,binaryResult],[1,2])

    resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)
    print(image_folder+"Segmented"+image_name.split('.')[0]+"_merge.jpg")
    cv2.imwrite(image_folder+"Segmented/"+image_name.split('.')[0]+"_merge.jpg",resultMerge)
    cv2.imwrite(image_folder+"Segmented/"+image_name.split('.')[0] + "_prob.bmp", (probResult*255).astype(np.uint8))