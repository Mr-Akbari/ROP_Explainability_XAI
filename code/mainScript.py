#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:12:50 2017

@author: veysiyildiz
"""

import opticDiscDetector as odd
import fasterTracing as tc
import minimumSpanningTree as mst
import cubicSpline as cs
import featureExtractionScript as fes
import pandas as pd
import numpy as np
import argparse
import generateScore as gs
import os
import os.path
import sys
import pickle
sys.path.insert(0,'code\segmentation')
import tensorflow as tf
#tf.python.control_flow_ops = tf
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from perception.infers.segmention_infer import vessel_segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Retinal Image Analysis Code ',\
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('pathToFolder', help='Path to folder which contains images and xlsx/csv file which contains names and disc centers (this folder should be placed in data folder)')
    parser.add_argument('imageNamesXlsFile', help='Name of the xlsx/csv file which contains names and disc centers') 
    parser.add_argument('scoreFileName', help='Name of the xlsx/csv File in which output scores are stored')
    parser.add_argument('--saveDebug',default=1 ,help="If you want to save the segmented image, features, centerline points of the vessel, this should be 1.")
    parser.add_argument('--featureFileName',default='Features.xlsx', help='Name of the xlsx/csv File in which output features are stored')
    parser.add_argument('--predictPlus',default=1, help='The system would predict image severity score in Plus or higher category by default. If you want to predict severity score in Normal category, this should be 0.')
    args = parser.parse_args()
       
    path= args.pathToFolder
    if path[-1] != "/":
        path+= "/"     
    path = '' + path
    #print("path:",path)
#    imageNames= args.imageNamesXlsFile
#    if imageNames[-5:] != ".xlsx":
#        imageNames=imageNames[:-5] + ".xlsx"
#  
#    if args.saveDebug:
#        featureFileName=args.featureFileName
#        if featureFileName[-5:] != ".xlsx":
#            featureFileName+= ".xlsx"
# 
    imageNames= args.imageNamesXlsFile
    if imageNames[-5:] != ".xlsx" and imageNames[-4:] != ".csv":
        print("imageNames file name must end with .xlsx or .csv")
        sys.exit()
  
    if args.saveDebug:
        featureFileName=args.featureFileName
        if featureFileName[-5:] != ".xlsx" and featureFileName[-4:] != ".csv":
            print("feature file name must end with .xlsx or .csv ")
            sys.exit()


#    scoreFileName = args.scoreFileName
#    if scoreFileName[-5:] != ".xlsx":
#        scoreFileName+= ".xlsx"

    scoreFileName = args.scoreFileName
    if scoreFileName[-5:] != ".xlsx" and scoreFileName[-4:] != ".csv":
        print("score file name must end with .xlsx or .csv ")
        sys.exit()
            
    isPlus = args.predictPlus 
    
    segmentationFileName= 'Segmented'  
    if imageNames[-5:] == ".xlsx":
        xl = pd.ExcelFile(path+imageNames)
        first_sheet = xl.parse(xl.sheet_names[0])
    else:
        first_sheet = pd.read_csv(path+imageNames, sep=',')

    #print(os.getcwd())   
    with open('parameters/featureList.txt','rb') as f:
        featNames= pickle.load(f)
        
    if args.saveDebug:            
        outputFeatureDf = pd.DataFrame([],columns=['Image Name']+ featNames )#'SegmentedImageName', 'Features'
        if featureFileName[-5:] == ".xlsx":
            featureWriter = pd.ExcelWriter(path+featureFileName, engine='xlsxwriter')
            outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1')
        else:
            outputFeatureDf.to_csv(path+featureFileName)

#    featureList=['DistanceToDiscCenter','CumulativeTortuosityIndex','IntegratedCurvature(IC)','IntegratedSquaredCurvature(ISC)'\
#                 ,'ICNormalizedbyChordLength','ICNormalizedbyCurveLength','ISCNormalizedbyChordLength','ISCNormalizedbyCurveLength','NormofAcceleration',\
#                 'Curvature','AverageSegmentDiameter','AveragePointDiameter']

    outputScoreDf = pd.DataFrame([],columns=['SegmentedImageName','Score'])
    if scoreFileName[-5:] == ".xlsx":
        scoreWriter = pd.ExcelWriter(path+scoreFileName, engine='xlsxwriter')
        outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1')
    else:
        outputScoreDf.to_csv(path+scoreFileName)



    # Process CNN Segmentations
    nonSegmentedImageIdxs=[idx for idx,i in enumerate(first_sheet['SegmentationName']) if i!=i]
    providedSegmentationIdxs = [idx for idx,i in enumerate(first_sheet['SegmentationName']) if i==i]
    print("nonSegmentedImageIdxs: ", nonSegmentedImageIdxs)
    print("providedSegmentationIdxs: ", providedSegmentationIdxs)
    #imgCNNPathList=[path+str(first_sheet['ImageName'][i]) for i in nonSegmentedImageIdxs]    
    #print("imgCNNPathList: ", imgCNNPathList)    
    cnnSegmentationsPath = path + segmentationFileName
    #print("cnnSegmentationsPath: ", cnnSegmentationsPath)    
    cnnModel = 'parameters/unet'
    print('---> Starting to vessel segmentation')
    
    for i in nonSegmentedImageIdxs:
        unetCNN = vessel_segmentation(str(first_sheet['ImageName'][i]),path)
    #CNNSegmentedImagesList, imgNamesDoneList, imgNameFailedList = unetCNN.segment_batch(imgCNNPathList)

    

    #CNNSegmentedImages=[i for i in CNNSegmentedImagesList]
    #fail_names = [i.split('/')[-1] for i in imgNameFailedList]
    #allSegmentedImageNames= list(first_sheet['SegmentationName'])
#    allSegmentedImageNames=[i if i==i else CNNSegmentedImages[] for idx,i in enumerate(allSegmentedImageNames)]                [nonSegmentedImageIdxs]=CNNSegmentedImages
    updated_segmentation_list = []
    for idx, image in enumerate(first_sheet['ImageName']):
        if first_sheet['SegmentationName'][idx] == first_sheet['SegmentationName'][idx]:
            updated_segmentation_list += [str(first_sheet['SegmentationName'][idx])]
        elif not str(image)[:-3] + 'png' in fail_names:
            updated_segmentation_list += [str(image)[:-3] + 'png']
        else:
            updated_segmentation_list += [None]
        
    print('---> Starting to finding Optic Disc')
    allSegmentedImageNames=list(map(str,updated_segmentation_list))
    for idx,segmentedImageName in enumerate(allSegmentedImageNames):
        print('Working on image: ', segmentedImageName)
        segmentedImagePath=path+segmentationFileName +'/'+ segmentedImageName
        origImageName = str(first_sheet['ImageName'][idx])
        #print("origImageName: ",origImageName)
        origImagePath = path + origImageName
        if os.path.isfile(segmentedImagePath):
            provided_center = np.array([first_sheet['CenterColumn'][idx],first_sheet['CenterRow'][idx]])
            if not all(np.isnan(provided_center)):
                cntr = provided_center
            else:
                cntr = odd.find_optic_disc_center(origImageName,path)
                print("cntr: ", cntr)
                outputcenterDf = pd.DataFrame({'CenterColumn': cntr[0],'CenterRow': cntr[1]}, index=[idx])
                outputcenterDf = pd.DataFrame(list(zip([cntr[0]],[cntr[1]])), columns=['CenterColumn','CenterRow'])#[cntr],columns=['CenterColumn','CenterRow'])
                print("outputcenterDf: ",outputcenterDf)
                print("path+imageNames: ",path+imageNames)
                centerWriter = pd.ExcelWriter(path+imageNames, engine='xlsxwriter')
                outputcenterDf.to_excel(centerWriter, sheet_name='Sheet1')

#            print cntr
            print('---> Disc center found!, finding the vessel tree information..')
            finalPoints, scatime,protime = tc.trace(segmentedImagePath,cntr, args.saveDebug)
            branches=mst.vesselTree(finalPoints,cntr)
            splines= cs.fitSplines(branches,finalPoints)  
            print('---> Starting to extract features..')            
            features=fes.extractFeatures(splines,segmentedImagePath,cntr)

            if args.saveDebug:
                tempDataFrame=pd.DataFrame([[segmentedImageName]+ list(features)], columns=['Image Name']+ featNames)
                outputFeatureDf=outputFeatureDf.append(tempDataFrame, ignore_index=True)
            score = gs.generateScore(features,isPlus)
            outputScoreDf=outputScoreDf.append(pd.DataFrame([[segmentedImageName,score]],columns=['SegmentedImageName','Score'] ), ignore_index=True)
        else:
            print(('Segmentation file of image "' + str(first_sheet['ImageName'][idx]) +
            '" could not be found! (Either segmentation code failed or segmanted image is not provided in the folder)'))

    if args.saveDebug: 
        if featureFileName[-5:] == ".xlsx":
            outputFeatureDf.to_excel(featureWriter, sheet_name='Sheet1',index=False)
            featureWriter.save()
        else:
            outputFeatureDf.to_csv(path+featureFileName, index=False)
            
    if scoreFileName[-5:] == ".xlsx":
        outputScoreDf.to_excel(scoreWriter, sheet_name='Sheet1',index=False)
        scoreWriter.sheets['Sheet1'].set_column('A:Z',20)
        scoreWriter.save()
    else:
        outputScoreDf.to_csv(path+scoreFileName)

 

    


        