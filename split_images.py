# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:04:02 2021

@author: Ningyu Wang
"""
import os, cv2
import cut
import numpy as np
from pathlib import Path

sourcepath = './median/result/seg'
targetpath = './split_after_seg'

dx = 156
Nx = 8
Dy = 223

from os.path import isfile, join
files = [f for f in os.listdir(sourcepath) if isfile(join(sourcepath, f))]

for ite in enumerate(files):
    if ite[1][-3:] == 'jpg' or ite[1][-3:] == 'tif':
        filename = sourcepath+'/'+ite[1];
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        print('create folder for '+ite[1][0:-4])
        #Path(targetpath).mkdir(parents=True,exist_ok=True)
        Path(targetpath+'/'+ite[1][0:-4]).mkdir(parents=True,exist_ok=True)
        '''
        try:
            #os.mkdir(targetpath)
            Path(targetpath).mkdir(parents=False,exist_ok=True)
        except FileExistsError:
            pass
        '''
        '''
        try:
            os.mkdir(targetpath+'/'+ite[1][0:2])
        except FileExistsError:
            pass
        '''
        for ite2 in range(Nx):

            '''
            try:
                os.mkdir(targetpath+'/'+ite[1][0:2]+'/'+str(ite2))

            except FileExistsError:
                pass
            '''
            x = [dx*ite2, dx*(ite2+1)]
            im_cut = cut.cut(im,x=[0,Dy],y=x,FlagPercent=False)
            savename = targetpath + '/' + ite[1][0:-4] + '/' + str(ite2)+'.tif'
            cv2.imwrite(savename,im_cut)
