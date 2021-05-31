# -*- coding: utf-8 -*-
"""
Created on Thu May 13 01:24:17 2021

compute the oil saturation from each histogram and segmented image

@author: Ningyu Wang
"""
import os, cv2
import cut
import numpy as np
from pathlib import Path
from imshow import imshow
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import time





# the average value of y
def yavg(x):    
    y = np.zeros(3);
    for i in range(3):
        y[i] = np.mean(x[:,:,1])
    #print(y)
    return(np.mean(y))

# Compute the oil saturation based on the segmented image,
# assuming that 255 is glass and oil while 0 is ferrofluid
# Returns the grayscale image and a 2D array of xy coordinates
# of points on the contour.
#
# Please notice that 
#   In cv2, 
#       x is top to bottom of the figure,
#       y is left to right of the figure
#   In others,
#       x is left to right of the figure
#       y is top to bottom of the figure

def findSo(filename,showresult=False):
    #filename = "test.jpg"
    print("Reading image : ", filename)
    imReference = cv2.imread(filename)
    imgray = imReference[:,:,2]



    # get the contours
    '''
    # The cv2.findContours() function removed the first output in a newer version.
    tmpim, contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
                                              mode=cv2.CHAIN_APPROX_SIMPLE)
    '''
    contours, hierarchy = cv2.findContours(imgray, method=cv2.RETR_TREE, \
                                       mode=cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(imgray, contours, contourIdx = -1, color=(255,0,0), \
#                     thickness=4)

    imshow(imgray,showresult,name='contours');
    

    # First, the real interfaces have a lot of points.
    # Abandon the contours with only several points.
    # Keep only the longest contours.
    contours = sorted(contours,key=len,reverse=True);
    #contoursorig = np.copy(contours)
    contours = contours[0:2];

    # Then, there are two longest contours found:
    # 1. the upper glass, average y value largest
    # 2. the lower glass, average y value smallest
    #   
    contours = sorted(contours,key=yavg)
    
    m,n = imgray.shape
    #print('m = ',m,'; n = ',n)
    x0 = contours[0][:,0,0]
    y0 = contours[0][:,0,1]
    x1 = contours[1][:,0,0]
    y1 = contours[1][:,0,1]
    
    # remove the points at the corners in the image
    for lp1 in range(len(x0)-1,-1,-1):
        #print(lp1)
        if y0[lp1] < 0.5 or y0[lp1] > m-1.5:
            x0 = np.delete(x0,lp1)
            y0 = np.delete(y0,lp1)
    for lp1 in range(len(x1)-1,-1,-1):
        #print(lp1)
        if y1[lp1] < 0.5 or y1[lp1] > m-1.5:
            x1 = np.delete(x1,lp1)
            y1 = np.delete(y1,lp1)
    # shift the array to make the first and last point on the left and right
    # margin of the image
    for lp1 in range(len(x0)):
        if (x0[0] < 0.5 and x0[-1] > n-1.5) or (x0[-1] < 0.5 and x0[0] > n-1.5):
            #print(x0)
            break
        else:
            x0 = np.roll(x0,1)
    for lp1 in range(len(x1)):
        #print('start = ',x1[0],'; end = ',x1[-1])
        if (x1[0] < 0.5 and x1[-1] > n-1.5) or (x1[-1] < 0.5 and x1[0] > n-1.5):
            #print(x1)
            break
        else:
            x1 = np.roll(x1,-1)

    
    x_channel = np.concatenate((x0,x1))
    y_channel = np.concatenate((y0,y1))

    
    # create a polygon of the flow channel
    polygon = np.array([(x_channel[x],y_channel[x]) for x in range(len(x_channel))],dtype=list)

    # find all points in the polygon
    path_polygon = pltPath.Path(polygon)
    points = np.array([[i,j] for i in range(0,n) for j in range(0,m)],dtype=list).astype('int')



    value = np.array([imgray[points[x,1],points[x,0]]>254.5 for x in range(len(points[:,0]))])
    
    inside = path_polygon.contains_points(points)
    area_channel = sum(inside)
    area_oil = sum(inside*value)
    
    So = area_oil / area_channel
    
    #So = 0

    

    return (imReference,contours,x0,y0,x1,y1,polygon,points,value,So)

def So_split_then_seg():
    
    path = "./split"

    path_seg = "seg"
    path_hist= "hist"
    
    # all folders in the split folder
    p = Path(path)
    dir_im = [x for x in p.iterdir() if x.is_dir()]
    
    # create a list to store the results
    result = []
    
    for ite in dir_im:
        result.append([ite.__str__(), []])
        
        # get all files in the segment folder
        dir_tmp = ite / path_seg;
        dir_seg = [x for x in dir_tmp.iterdir() if ~x.is_dir()]
        #print(dir_seg)
        for ite2 in dir_seg:
            #print(dir_seg[ite2])
            filename = ite2.__str__()
            if filename[-4:] != '.tif':
                continue
            imgray, contours, x0,y0, x1,y1, polygon,points,v,So = findSo(filename)
            #imshow(imgray,name=filename)
            result[-1][-1].append(So)
            
        

if __name__ == '__main__':
    path = "./split_after_seg"

    path_seg = "seg"
    path_hist= "hist"
    
    # all folders in the split folder
    p = Path(path)
    dir_im = [x for x in p.iterdir() if x.is_dir()]
    
    # create a list to store the results
    result = []
    
    for ite in dir_im:
        result.append([ite.__str__(), []])
        
        # get all files in the segment folder
        dir_tmp = ite ;
        dir_seg = [x for x in dir_tmp.iterdir() if ~x.is_dir()]
        #print(dir_seg)
        for ite2 in dir_seg:
            #print(dir_seg[ite2])
            filename = ite2.__str__()
            if filename[-4:] != '.tif':
                continue
            imgray, contours, x0,y0, x1,y1, polygon,points,v,So = findSo(filename)
            #imshow(imgray,name=filename)
            result[-1][-1].append(So)
            cv2.drawContours(imgray, contours, -1, (255,0,0), 4)
            imshow(imgray)
            time.sleep(1)
            cv2.destroyAllWindows()
            
            
    